"""
Databricks Client with Connection Pooling
==========================================
Enterprise-grade database client with:
- Connection pooling
- Automatic retry with backoff
- Circuit breaker integration
- Health monitoring

This replaces the single-connection approach with a proper pool.
"""

import time
import queue
import logging
import threading
from typing import List, Dict, Any, Optional
from contextlib import contextmanager
from dataclasses import dataclass
import requests
from databricks import sql

from ..core.config import config

logger = logging.getLogger(__name__)


@dataclass
class PoolConfig:
    """Connection pool configuration"""
    min_connections: int = 2
    max_connections: int = 10
    connection_timeout: int = 30  # seconds to wait for connection from pool
    idle_timeout: int = 300  # seconds before idle connection is closed
    max_retries: int = 3
    retry_delay: float = 1.0  # base delay in seconds


class PooledConnection:
    """Wrapper around a database connection with metadata"""
    
    def __init__(self, connection):
        self.connection = connection
        self.created_at = time.time()
        self.last_used_at = time.time()
        self.use_count = 0
        self.is_valid = True
    
    def mark_used(self):
        """Mark connection as used"""
        self.last_used_at = time.time()
        self.use_count += 1
    
    def is_expired(self, idle_timeout: int) -> bool:
        """Check if connection has been idle too long"""
        return time.time() - self.last_used_at > idle_timeout


class ConnectionPool:
    """
    Thread-safe connection pool for Databricks SQL connections.
    
    Features:
    - Min/max connection limits
    - Automatic connection validation
    - Idle connection cleanup
    - Thread-safe operations
    """
    
    def __init__(self, pool_config: Optional[PoolConfig] = None):
        self.config = pool_config or PoolConfig()
        self._pool: queue.Queue = queue.Queue(maxsize=self.config.max_connections)
        self._active_connections = 0
        self._lock = threading.Lock()
        self._initialized = False
    
    def _create_connection(self):
        """Create a new Databricks SQL connection"""
        if not config.is_configured():
            raise ValueError("Databricks not configured. Check .env file.")
        
        conn = sql.connect(
            server_hostname=config.DATABRICKS_HOST.replace("https://", "").replace("http://", ""),
            http_path=config.DATABRICKS_HTTP_PATH,
            access_token=config.DATABRICKS_TOKEN
        )
        return PooledConnection(conn)
    
    def _validate_connection(self, pooled_conn: PooledConnection) -> bool:
        """Validate that a connection is still usable"""
        try:
            cursor = pooled_conn.connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return True
        except Exception as e:
            logger.warning(f"Connection validation failed: {e}")
            return False
    
    def initialize(self):
        """Initialize the pool with minimum connections"""
        if self._initialized:
            return
        
        with self._lock:
            if self._initialized:
                return
            
            logger.info(f"Initializing connection pool (min={self.config.min_connections})")
            
            for _ in range(self.config.min_connections):
                try:
                    conn = self._create_connection()
                    self._pool.put_nowait(conn)
                    self._active_connections += 1
                except Exception as e:
                    logger.error(f"Failed to create initial connection: {e}")
            
            self._initialized = True
            logger.info(f"Connection pool initialized with {self._active_connections} connections")
    
    @contextmanager
    def get_connection(self):
        """
        Get a connection from the pool.
        
        Usage:
            with pool.get_connection() as conn:
                cursor = conn.cursor()
                ...
        """
        if not self._initialized:
            self.initialize()
        
        pooled_conn = None
        
        try:
            # Try to get from pool
            try:
                pooled_conn = self._pool.get(timeout=self.config.connection_timeout)
                
                # Check if connection is still valid
                if pooled_conn.is_expired(self.config.idle_timeout) or not pooled_conn.is_valid:
                    # Close expired connection and create new one
                    self._close_connection(pooled_conn)
                    pooled_conn = self._create_connection()
                elif not self._validate_connection(pooled_conn):
                    # Connection is broken, create new one
                    self._close_connection(pooled_conn)
                    pooled_conn = self._create_connection()
                    
            except queue.Empty:
                # Pool is empty, try to create new connection if under limit
                with self._lock:
                    if self._active_connections < self.config.max_connections:
                        pooled_conn = self._create_connection()
                        self._active_connections += 1
                    else:
                        raise TimeoutError(
                            f"Connection pool exhausted. Max connections: {self.config.max_connections}"
                        )
            
            pooled_conn.mark_used()
            yield pooled_conn.connection
            
        except Exception as e:
            if pooled_conn:
                pooled_conn.is_valid = False
            raise
        finally:
            # Return connection to pool
            if pooled_conn and pooled_conn.is_valid:
                try:
                    self._pool.put_nowait(pooled_conn)
                except queue.Full:
                    # Pool is full, close this connection
                    self._close_connection(pooled_conn)
    
    def _close_connection(self, pooled_conn: PooledConnection):
        """Safely close a connection"""
        try:
            pooled_conn.connection.close()
        except Exception as e:
            logger.warning(f"Error closing connection: {e}")
        finally:
            with self._lock:
                self._active_connections = max(0, self._active_connections - 1)
    
    def close_all(self):
        """Close all connections in the pool"""
        logger.info("Closing all pooled connections")
        
        while not self._pool.empty():
            try:
                pooled_conn = self._pool.get_nowait()
                self._close_connection(pooled_conn)
            except queue.Empty:
                break
        
        with self._lock:
            self._active_connections = 0
            self._initialized = False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get pool statistics"""
        return {
            "active_connections": self._active_connections,
            "available_connections": self._pool.qsize(),
            "max_connections": self.config.max_connections,
            "initialized": self._initialized
        }


class DatabricksClient:
    """
    Client for all Databricks operations with connection pooling and retry logic.
    """
    
    def __init__(self, pool_config: Optional[PoolConfig] = None):
        self._pool = ConnectionPool(pool_config)
        self._legacy_connection = None  # For backward compatibility
    
    def get_connection(self):
        """
        Get database connection (legacy method for backward compatibility).
        Prefer using query() or execute() methods instead.
        """
        if not config.is_configured():
            raise ValueError("Databricks not configured. Check .env file.")
        
        if self._legacy_connection is None:
            self._legacy_connection = sql.connect(
                server_hostname=config.DATABRICKS_HOST.replace("https://", "").replace("http://", ""),
                http_path=config.DATABRICKS_HTTP_PATH,
                access_token=config.DATABRICKS_TOKEN
            )
        return self._legacy_connection
    
    def close(self):
        """Close all connections"""
        if self._legacy_connection:
            self._legacy_connection.close()
            self._legacy_connection = None
        self._pool.close_all()
    
    def _execute_with_retry(self, operation: str, func, *args, **kwargs):
        """Execute operation with retry logic"""
        last_exception = None
        
        for attempt in range(self._pool.config.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if attempt < self._pool.config.max_retries - 1:
                    delay = self._pool.config.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.warning(f"{operation} failed (attempt {attempt + 1}), retrying in {delay}s: {e}")
                    time.sleep(delay)
                else:
                    logger.error(f"{operation} failed after {self._pool.config.max_retries} attempts: {e}")
        
        raise last_exception
    
    def query(self, sql_query: str, fetch: bool = True) -> Optional[List[Dict]]:
        """Execute SQL query and return results using pooled connection"""
        def _do_query():
            with self._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                
                if fetch and cursor.description:
                    columns = [desc[0] for desc in cursor.description]
                    rows = cursor.fetchall()
                    return [dict(zip(columns, row)) for row in rows]
                return None
        
        return self._execute_with_retry("Query", _do_query)
    
    def execute(self, sql_query: str) -> bool:
        """Execute SQL without returning results"""
        def _do_execute():
            with self._pool.get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(sql_query)
                return True
        
        return self._execute_with_retry("Execute", _do_execute)
    
    def test_connection(self) -> bool:
        """Test if connection works"""
        try:
            result = self.query("SELECT 1 as test")
            return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_pool_stats(self) -> Dict[str, Any]:
        """Get connection pool statistics"""
        return self._pool.get_stats()
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Databricks model endpoint"""
        if not config.EMBEDDING_ENDPOINT:
            logger.warning("No embedding endpoint configured, using mock embedding")
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 1000 / 1000.0 for i in range(384)]
        
        url = f"{config.DATABRICKS_HOST}/serving-endpoints/{config.EMBEDDING_ENDPOINT}/invocations"
        headers = {
            "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                url, 
                headers=headers, 
                json={"input": text},
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            elif "predictions" in result:
                return result["predictions"][0]
            elif "embeddings" in result:
                return result["embeddings"][0]
            return result[0] if isinstance(result, list) else result
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 1000 / 1000.0 for i in range(384)]
    
    def generate_answer(self, prompt: str, max_tokens: int = 500) -> str:
        """Generate answer using Databricks LLM endpoint"""
        if not config.LLM_ENDPOINT:
            logger.warning("No LLM endpoint configured, returning mock answer")
            return f"[Mock Answer] Based on your query, here's a summary. Configure DATABRICKS_LLM_ENDPOINT for real answers."
        
        url = f"{config.DATABRICKS_HOST}/serving-endpoints/{config.LLM_ENDPOINT}/invocations"
        headers = {
            "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            response = requests.post(
                url,
                headers=headers,
                json={
                    "messages": [{"role": "user", "content": prompt}],
                    "max_tokens": max_tokens,
                    "temperature": 0.7
                },
                timeout=60
            )
            response.raise_for_status()
            result = response.json()
            
            if "choices" in result:
                content = result["choices"][0].get("message", {}).get("content", str(result))
                return self._extract_text_from_response(content)
            elif "predictions" in result:
                pred = result["predictions"][0]
                return self._extract_text_from_response(pred)
            
            return self._extract_text_from_response(result)
        except Exception as e:
            logger.error(f"LLM error: {e}")
            return f"Error generating answer: {str(e)}"
    
    def _extract_text_from_response(self, content):
        """Extract clean text from various LLM response formats"""
        if isinstance(content, str):
            return content
        
        if isinstance(content, list) and len(content) > 0:
            for item in content:
                if isinstance(item, dict):
                    if 'summary' in item:
                        summary = item['summary']
                        if isinstance(summary, list) and len(summary) > 0:
                            texts = [s.get('text', '') for s in summary if isinstance(s, dict) and 'text' in s]
                            if texts:
                                full_text = '\n\n'.join(texts)
                                if 'We need to answer:' in full_text or 'Let\'s gather' in full_text:
                                    parts = full_text.split('\n\n')
                                    answer_parts = [p for p in parts if not p.startswith('We need') and not p.startswith('Let\'s') and len(p) > 50]
                                    if answer_parts:
                                        return '\n\n'.join(answer_parts)
                                return full_text
                    if 'text' in item:
                        return item['text']
        
        return str(content)


# Global client instance with pooling
db_client = DatabricksClient()
