"""Analytics Module - Data handling and query processing"""

from .data_handler import (
    process_churn_file,
    get_churn_data,
    get_churn_stats,
    get_churn_by_category,
    get_table_schema
)
from .text_to_sql import TextToSQLAgent
from .query_router import QueryRouter, QueryType

# Create singleton instances
text_to_sql_agent = TextToSQLAgent()
query_router = QueryRouter()

__all__ = [
    "process_churn_file",
    "get_churn_data",
    "get_churn_stats",
    "get_churn_by_category",
    "get_table_schema",
    "TextToSQLAgent",
    "text_to_sql_agent",
    "QueryRouter",
    "query_router",
    "QueryType"
]
