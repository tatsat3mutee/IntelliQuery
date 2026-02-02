"""
Configuration - All settings from environment variables
"""

import os
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Application configuration - Databricks only"""
    
    # Databricks Connection
    DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")
    DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN", "")
    DATABRICKS_HTTP_PATH = os.getenv("DATABRICKS_HTTP_PATH", "")
    
    # Schema
    DATABRICKS_CATALOG = os.getenv("DATABRICKS_CATALOG", "main")
    DATABRICKS_SCHEMA = os.getenv("DATABRICKS_SCHEMA", "intelliquery_tatsat")
    
    # Model Endpoints (your hosted models in Databricks)
    LLM_ENDPOINT = os.getenv("DATABRICKS_LLM_ENDPOINT", "")
    EMBEDDING_ENDPOINT = os.getenv("DATABRICKS_EMBEDDING_ENDPOINT", "")
    
    # Vector Search Settings
    VECTOR_SEARCH_ENDPOINT = os.getenv("DATABRICKS_VECTOR_SEARCH_ENDPOINT", "")
    USE_VECTOR_SEARCH_INDEX = os.getenv("USE_VECTOR_SEARCH_INDEX", "false").lower() == "true"
    
    # Table names (2 tables!)
    @property
    def RAG_TABLE(self) -> str:
        return f"{self.DATABRICKS_CATALOG}.{self.DATABRICKS_SCHEMA}.rag_documents"
    
    @property
    def CHURN_TABLE(self) -> str:
        return f"{self.DATABRICKS_CATALOG}.{self.DATABRICKS_SCHEMA}.telco_churn_data"
    
    @property
    def VECTOR_INDEX_NAME(self) -> str:
        """Full vector search index name"""
        return f"{self.DATABRICKS_CATALOG}.{self.DATABRICKS_SCHEMA}.rag_documents_index"
    
    # App Settings
    MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
    
    def is_configured(self) -> bool:
        """Check if Databricks is configured"""
        return bool(self.DATABRICKS_HOST and self.DATABRICKS_TOKEN and self.DATABRICKS_HTTP_PATH)
    
    def get_status(self) -> dict:
        """Get configuration status"""
        return {
            "databricks_configured": self.is_configured(),
            "host": self.DATABRICKS_HOST[:30] + "..." if self.DATABRICKS_HOST else "Not set",
            "catalog": self.DATABRICKS_CATALOG,
            "schema": self.DATABRICKS_SCHEMA,
            "rag_table": self.RAG_TABLE,
            "churn_table": self.CHURN_TABLE,
            "llm_endpoint": self.LLM_ENDPOINT or "Not set",
            "embedding_endpoint": self.EMBEDDING_ENDPOINT or "Not set",
            "vector_search_endpoint": self.VECTOR_SEARCH_ENDPOINT or "Not set",
            "use_vector_index": self.USE_VECTOR_SEARCH_INDEX,
            "vector_index_name": self.VECTOR_INDEX_NAME
        }


# Global config instance
config = Config()
