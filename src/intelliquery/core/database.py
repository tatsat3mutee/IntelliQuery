"""
Databricks Client - Handles all Databricks interactions
- SQL queries
- Embedding endpoint
- LLM endpoint
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from databricks import sql

from ..core.config import config

logger = logging.getLogger(__name__)


class DatabricksClient:
    """Client for all Databricks operations"""
    
    def __init__(self):
        self._connection = None
    
    def get_connection(self):
        """Get database connection (reuse if possible)"""
        if not config.is_configured():
            raise ValueError("Databricks not configured. Check .env file.")
        
        if self._connection is None:
            self._connection = sql.connect(
                server_hostname=config.DATABRICKS_HOST.replace("https://", "").replace("http://", ""),
                http_path=config.DATABRICKS_HTTP_PATH,
                access_token=config.DATABRICKS_TOKEN
            )
        return self._connection
    
    def close(self):
        """Close connection"""
        if self._connection:
            self._connection.close()
            self._connection = None
    
    def query(self, sql_query: str, fetch: bool = True) -> Optional[List[Dict]]:
        """Execute SQL query and return results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)
            
            if fetch and cursor.description:
                columns = [desc[0] for desc in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
            return None
        except Exception as e:
            logger.error(f"Query error: {e}")
            raise
    
    def execute(self, sql_query: str) -> bool:
        """Execute SQL without returning results"""
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            cursor.execute(sql_query)
            return True
        except Exception as e:
            logger.error(f"Execute error: {e}")
            raise
    
    def test_connection(self) -> bool:
        """Test if connection works"""
        try:
            result = self.query("SELECT 1 as test")
            return result is not None
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            return False
    
    def get_embedding(self, text: str) -> List[float]:
        """Get embedding from Databricks model endpoint"""
        if not config.EMBEDDING_ENDPOINT:
            logger.warning("No embedding endpoint configured, using mock embedding")
            # Return mock embedding (384 dimensions)
            import hashlib
            hash_val = int(hashlib.md5(text.encode()).hexdigest(), 16)
            return [(hash_val >> i) % 1000 / 1000.0 for i in range(384)]
        
        url = f"{config.DATABRICKS_HOST}/serving-endpoints/{config.EMBEDDING_ENDPOINT}/invocations"
        headers = {
            "Authorization": f"Bearer {config.DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        try:
            # Databricks Foundation Model API format
            response = requests.post(
                url, 
                headers=headers, 
                json={"input": text},  # Single text, not array
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            
            # Handle different response formats
            if "data" in result and len(result["data"]) > 0:
                return result["data"][0]["embedding"]
            elif "predictions" in result:
                return result["predictions"][0]
            elif "embeddings" in result:
                return result["embeddings"][0]
            return result[0] if isinstance(result, list) else result
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            # Return mock embedding on failure
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
            
            logger.info(f"LLM Response keys: {result.keys() if isinstance(result, dict) else type(result)}")
            
            # Handle different response formats
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
        # If it's already a string, return it
        if isinstance(content, str):
            return content
        
        # Handle list responses with reasoning/summary structure
        if isinstance(content, list) and len(content) > 0:
            for item in content:
                if isinstance(item, dict):
                    # Look for summary text
                    if 'summary' in item:
                        summary = item['summary']
                        if isinstance(summary, list) and len(summary) > 0:
                            # Get all text from summary items
                            texts = [s.get('text', '') for s in summary if isinstance(s, dict) and 'text' in s]
                            if texts:
                                # Return all text, skip internal reasoning
                                full_text = '\n\n'.join(texts)
                                # If it looks like reasoning, extract just the answer part
                                if 'We need to answer:' in full_text or 'Let\'s gather' in full_text:
                                    # Try to find the actual answer after reasoning
                                    parts = full_text.split('\n\n')
                                    # Skip reasoning parts, get substantive answer
                                    answer_parts = [p for p in parts if not p.startswith('We need') and not p.startswith('Let\'s') and len(p) > 50]
                                    if answer_parts:
                                        return '\n\n'.join(answer_parts)
                                return full_text
                    # Look for text field directly
                    if 'text' in item:
                        return item['text']
        
        # Fallback: convert to string
        return str(content)


# Global client instance
db_client = DatabricksClient()
