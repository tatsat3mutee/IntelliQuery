"""
Databricks Vector Search Integration
====================================
High-performance semantic search using Databricks Vector Search Index

Features:
- Automatic index creation and management
- Sub-second search on millions of documents
- ANN (Approximate Nearest Neighbor) algorithm
- Fallback to in-memory search if index not available
"""

import logging
from typing import List, Dict, Optional

from ..core.config import config

logger = logging.getLogger(__name__)


class VectorSearchManager:
    """Manages Databricks Vector Search operations"""
    
    def __init__(self):
        self.client = None
        self.index_name = config.VECTOR_INDEX_NAME
        self.endpoint_name = config.VECTOR_SEARCH_ENDPOINT
        self._index_exists = None  # Cache index existence check
    
    def _get_client(self):
        """Get or create Vector Search client"""
        if self.client is None:
            try:
                from databricks.vector_search.client import VectorSearchClient
                
                # Initialize client with workspace URL
                workspace_url = config.DATABRICKS_HOST
                if not workspace_url.startswith('https://'):
                    workspace_url = f"https://{workspace_url}"
                
                self.client = VectorSearchClient(
                    workspace_url=workspace_url,
                    personal_access_token=config.DATABRICKS_TOKEN
                )
                logger.info(f"✓ Vector Search client initialized for {workspace_url}")
            except ImportError:
                logger.warning("⚠️ databricks-vectorsearch not installed. Run: pip install databricks-vectorsearch")
                return None
            except Exception as e:
                logger.error(f"❌ Failed to initialize Vector Search client: {e}")
                return None
        
        return self.client
    
    def is_available(self) -> bool:
        """Check if Vector Search is configured and available"""
        return bool(
            config.USE_VECTOR_SEARCH_INDEX and 
            config.VECTOR_SEARCH_ENDPOINT and 
            self._get_client() is not None
        )
    
    def index_exists(self) -> bool:
        """Check if vector search index exists"""
        if self._index_exists is not None:
            return self._index_exists
        
        client = self._get_client()
        if not client:
            self._index_exists = False
            return False
        
        try:
            # Try to get index info
            client.get_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            )
            self._index_exists = True
            logger.info(f"✓ Vector index exists: {self.index_name}")
            return True
        except Exception as e:
            logger.info(f"Index not found: {e}")
            self._index_exists = False
            return False
    
    def create_index(
        self,
        embedding_dimension: int = 384,
        distance_metric: str = "COSINE",
        sync_mode: str = "CONTINUOUS"
    ) -> Dict:
        """
        Create vector search index on existing Delta table
        
        Args:
            embedding_dimension: Size of embedding vectors (default 384 for BGE/MiniLM)
            distance_metric: "COSINE", "EUCLIDEAN", or "DOT_PRODUCT"
            sync_mode: "CONTINUOUS" (auto-sync) or "TRIGGERED" (manual sync)
        
        Returns:
            Dict with success status and message
        """
        try:
            client = self._get_client()
            if not client:
                return {
                    "success": False,
                    "error": "Vector Search client not available"
                }
            
            if not self.endpoint_name:
                return {
                    "success": False,
                    "error": "Vector Search endpoint not configured. Set DATABRICKS_VECTOR_SEARCH_ENDPOINT"
                }
            
            # Check if endpoint exists, create if not
            try:
                client.get_endpoint(self.endpoint_name)
                logger.info(f"✓ Endpoint exists: {self.endpoint_name}")
            except:
                logger.info(f"Creating endpoint: {self.endpoint_name}")
                client.create_endpoint(
                    name=self.endpoint_name,
                    endpoint_type="STANDARD"  # or "PERFORMANCE_OPTIMIZED"
                )
            
            # Create Delta Sync Index (syncs from Delta table)
            logger.info(f"Creating vector index: {self.index_name}")
            logger.info(f"  Source table: {config.RAG_TABLE}")
            logger.info(f"  Embedding dimension: {embedding_dimension}")
            logger.info(f"  Distance metric: {distance_metric}")
            logger.info(f"  Sync mode: {sync_mode}")
            
            # Try CONTINUOUS first, fallback to TRIGGERED if not supported
            try:
                index = client.create_delta_sync_index(
                    endpoint_name=self.endpoint_name,
                    index_name=self.index_name,
                    source_table_name=config.RAG_TABLE,
                    pipeline_type=sync_mode,
                    primary_key="id",
                    embedding_dimension=embedding_dimension,
                    embedding_vector_column="embedding",
                    columns_to_sync=["id", "filename", "text", "embedding", "chunk_index", "upload_date"]
                )
            except Exception as e:
                error_msg = str(e)
                
                # Handle CONTINUOUS mode not supported
                if "CONTINUOUS is not supported" in error_msg and sync_mode == "CONTINUOUS":
                    logger.warning("⚠️ CONTINUOUS sync not supported, trying TRIGGERED mode...")
                    sync_mode = "TRIGGERED"
                    try:
                        index = client.create_delta_sync_index(
                            endpoint_name=self.endpoint_name,
                            index_name=self.index_name,
                            source_table_name=config.RAG_TABLE,
                            pipeline_type=sync_mode,
                            primary_key="id",
                            embedding_dimension=embedding_dimension,
                            embedding_vector_column="embedding",
                            columns_to_sync=["id", "filename", "text", "embedding", "chunk_index", "upload_date"]
                        )
                    except Exception as e2:
                        error_msg2 = str(e2)
                        # Handle Change Data Feed not enabled
                        if "change data feed" in error_msg2.lower() or "enableChangeDataFeed" in error_msg2:
                            logger.warning(f"⚠️ Change Data Feed not enabled on {config.RAG_TABLE}, enabling it now...")
                            self._enable_change_data_feed(config.RAG_TABLE)
                            logger.info("✓ Change Data Feed enabled, retrying index creation...")
                            index = client.create_delta_sync_index(
                                endpoint_name=self.endpoint_name,
                                index_name=self.index_name,
                                source_table_name=config.RAG_TABLE,
                                pipeline_type=sync_mode,
                                primary_key="id",
                                embedding_dimension=embedding_dimension,
                                embedding_vector_column="embedding",
                                columns_to_sync=["id", "filename", "text", "embedding", "chunk_index", "upload_date"]
                            )
                        else:
                            raise
                # Handle Change Data Feed not enabled (for CONTINUOUS mode)
                elif "change data feed" in error_msg.lower() or "enableChangeDataFeed" in error_msg:
                    logger.warning(f"⚠️ Change Data Feed not enabled on {config.RAG_TABLE}, enabling it now...")
                    self._enable_change_data_feed(config.RAG_TABLE)
                    logger.info("✓ Change Data Feed enabled, retrying index creation...")
                    index = client.create_delta_sync_index(
                        endpoint_name=self.endpoint_name,
                        index_name=self.index_name,
                        source_table_name=config.RAG_TABLE,
                        pipeline_type=sync_mode,
                        primary_key="id",
                        embedding_dimension=embedding_dimension,
                        embedding_vector_column="embedding",
                        columns_to_sync=["id", "filename", "text", "embedding", "chunk_index", "upload_date"]
                    )
                else:
                    raise
            
            self._index_exists = True
            
            logger.info(f"✅ Vector index created successfully!")
            logger.info(f"   Index: {self.index_name}")
            logger.info(f"   Status: {sync_mode} sync enabled")
            
            if sync_mode == "TRIGGERED":
                logger.info(f"⚠️  MANUAL SYNC REQUIRED!")
                logger.info(f"   After uploading documents, run:")
                logger.info(f"   curl -X POST http://localhost:8000/vector-search/sync")
            
            return {
                "success": True,
                "message": f"Vector index created: {self.index_name}",
                "index_name": self.index_name,
                "endpoint": self.endpoint_name,
                "sync_mode": sync_mode,
                "status": "syncing" if sync_mode == "CONTINUOUS" else "ready",
                "manual_sync_required": sync_mode == "TRIGGERED"
            }
        
        except Exception as e:
            logger.error(f"❌ Failed to create index: {e}", exc_info=True)
            return {
                "success": False,
                "error": str(e)
            }
    
    def _enable_change_data_feed(self, table_name: str):
        """Enable Change Data Feed on a Delta table"""
        try:
            from intelliquery.core.database import DatabaseClient
            db_client = DatabaseClient()
            
            sql = f"ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            logger.info(f"Executing: {sql}")
            db_client.execute_query(sql)
            logger.info(f"✓ Change Data Feed enabled on {table_name}")
        except Exception as e:
            logger.error(f"Failed to enable Change Data Feed: {e}")
            raise Exception(
                f"Could not enable Change Data Feed on {table_name}. "
                f"Please run manually: ALTER TABLE {table_name} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
            )
    
    def sync_index(self) -> Dict:
        """
        Manually trigger index sync (for TRIGGERED sync mode)
        
        Returns:
            Dict with sync status
        """
        try:
            client = self._get_client()
            if not client:
                return {"success": False, "error": "Client not available"}
            
            if not self.index_exists():
                return {"success": False, "error": "Index does not exist"}
            
            logger.info(f"Triggering manual sync for {self.index_name}")
            client.sync_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            )
            
            return {
                "success": True,
                "message": "Index sync triggered",
                "index_name": self.index_name
            }
        
        except Exception as e:
            logger.error(f"Sync error: {e}")
            return {"success": False, "error": str(e)}
    
    def search(
        self,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict] = None
    ) -> List[Dict]:
        """
        Search vector index for similar documents
        
        Args:
            query_vector: Embedding vector of the query
            top_k: Number of results to return
            filters: Optional filters (e.g., {"filename": "doc.pdf"})
        
        Returns:
            List of matching documents with similarity scores
        """
        try:
            client = self._get_client()
            if not client:
                raise Exception("Vector Search client not available")
            
            if not self.index_exists():
                raise Exception("Vector index does not exist. Create it first.")
            
            # Build search parameters
            search_params = {
                "columns": ["id", "filename", "text", "chunk_index", "upload_date"],
                "num_results": top_k
            }
            
            # Add filters if provided
            if filters:
                # Convert to Databricks filter format
                filter_str = " AND ".join([f"{k} = '{v}'" for k, v in filters.items()])
                search_params["filters"] = filter_str
            
            # Execute search
            logger.info(f"Searching index {self.index_name} for top {top_k} results")
            
            results = client.similarity_search(
                index_name=self.index_name,
                query_vector=query_vector,
                columns=search_params["columns"],
                num_results=top_k,
                filters=search_params.get("filters")
            )
            
            # Convert results to our format
            documents = []
            if results and 'result' in results and 'data_array' in results['result']:
                for row in results['result']['data_array']:
                    # Parse row based on column order
                    doc = {
                        'id': row[0],
                        'filename': row[1],
                        'text': row[2],
                        'chunk_index': row[3] if len(row) > 3 else 0,
                        'upload_date': row[4] if len(row) > 4 else None,
                        'similarity': row[-1] if len(row) > 5 else 1.0  # Score in last column
                    }
                    documents.append(doc)
            
            logger.info(f"✓ Found {len(documents)} documents from vector index")
            return documents
        
        except Exception as e:
            logger.error(f"Vector search error: {e}", exc_info=True)
            raise
    
    def get_index_info(self) -> Dict:
        """Get information about the vector index"""
        try:
            client = self._get_client()
            if not client:
                return {"success": False, "error": "Client not available"}
            
            if not self.index_exists():
                return {
                    "success": False,
                    "error": "Index does not exist",
                    "index_name": self.index_name
                }
            
            index_info = client.get_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            )
            
            return {
                "success": True,
                "index_name": self.index_name,
                "endpoint": self.endpoint_name,
                "status": index_info.get("status", {}).get("state", "unknown"),
                "source_table": config.RAG_TABLE,
                "info": index_info
            }
        
        except Exception as e:
            logger.error(f"Error getting index info: {e}")
            return {"success": False, "error": str(e)}
    
    def delete_index(self) -> Dict:
        """Delete the vector search index"""
        try:
            client = self._get_client()
            if not client:
                return {"success": False, "error": "Client not available"}
            
            logger.warning(f"Deleting vector index: {self.index_name}")
            client.delete_index(
                endpoint_name=self.endpoint_name,
                index_name=self.index_name
            )
            
            self._index_exists = False
            
            return {
                "success": True,
                "message": f"Index deleted: {self.index_name}"
            }
        
        except Exception as e:
            logger.error(f"Delete error: {e}")
            return {"success": False, "error": str(e)}


# Global instance
vector_search_manager = VectorSearchManager()
