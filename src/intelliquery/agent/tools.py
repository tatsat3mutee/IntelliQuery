"""
Tool Registry - Centralized Tool Management
===========================================
Wraps existing IntelliQuery modules as callable tools for the agent system.

Each tool is:
- Self-describing (for planner prompts)
- Independently executable
- Error-handled
- Stateless (state is managed by AgentState)
"""

import logging
from typing import Dict, List, Any, Optional, Callable
from functools import wraps

from .base import Tool, ToolCategory

logger = logging.getLogger(__name__)


def tool_error_handler(func: Callable) -> Callable:
    """Decorator to standardize tool error handling"""
    @wraps(func)
    def wrapper(*args, **kwargs) -> Dict[str, Any]:
        try:
            result = func(*args, **kwargs)
            # Ensure result is a dict with success field
            if isinstance(result, dict):
                if "success" not in result:
                    result["success"] = True
                return result
            return {"success": True, "result": result}
        except Exception as e:
            logger.error(f"Tool error in {func.__name__}: {e}")
            return {"success": False, "error": str(e)}
    return wrapper


class ToolRegistry:
    """
    Central registry for all available tools.
    
    Provides:
    - Tool registration and lookup
    - Tool descriptions for planner
    - Tool execution with error handling
    """
    
    def __init__(self):
        self._tools: Dict[str, Tool] = {}
        self._initialized = False
    
    def register(self, tool: Tool):
        """Register a tool in the registry"""
        self._tools[tool.name] = tool
        logger.info(f"Registered tool: {tool.name} ({tool.category.value})")
    
    def get(self, name: str) -> Optional[Tool]:
        """Get a tool by name"""
        return self._tools.get(name)
    
    def list_tools(self, category: Optional[ToolCategory] = None) -> List[Tool]:
        """List all tools, optionally filtered by category"""
        if category:
            return [t for t in self._tools.values() if t.category == category]
        return list(self._tools.values())
    
    def get_tool_descriptions(self) -> str:
        """Generate tool descriptions for planner prompt"""
        descriptions = []
        
        # Group by category
        categories = {}
        for tool in self._tools.values():
            cat = tool.category.value
            if cat not in categories:
                categories[cat] = []
            categories[cat].append(tool)
        
        for cat_name, tools in categories.items():
            descriptions.append(f"\n## {cat_name.upper()} Tools:")
            for tool in tools:
                descriptions.append(tool.to_prompt_description())
        
        return "\n".join(descriptions)
    
    def execute_tool(self, name: str, **kwargs) -> Dict[str, Any]:
        """Execute a tool by name with given parameters"""
        tool = self.get(name)
        if not tool:
            return {"success": False, "error": f"Tool '{name}' not found"}
        return tool.execute(**kwargs)
    
    def initialize_tools(self):
        """Initialize all available tools - called on first use"""
        if self._initialized:
            return
        
        # Import modules here to avoid circular imports
        from ..rag.document_processor import (
            search_documents, 
            answer_question,
            get_document_stats
        )
        from ..analytics.text_to_sql import text_to_sql_agent
        from ..analytics.data_handler import (
            get_churn_stats,
            get_churn_by_category,
            get_churn_data
        )
        from ..ml.predictor import churn_predictor
        from ..visualization.chart_generator import (
            generate_churn_distribution_chart,
            generate_churn_by_category_chart,
            generate_feature_importance_chart,
            generate_risk_distribution_chart,
            generate_prediction_comparison_chart
        )
        
        # ============== RAG TOOLS ==============
        
        self.register(Tool(
            name="rag_search",
            description="Search uploaded documents for relevant information. Returns top matching text chunks.",
            category=ToolCategory.RAG,
            function=tool_error_handler(search_documents),
            parameters={"question": "str - The search query", "top_k": "int - Number of results (default: 5)"},
            returns={"chunks": "List of matching document chunks with similarity scores"}
        ))
        
        self.register(Tool(
            name="rag_answer",
            description="Answer a question using uploaded documents. Uses RAG to find context and generate answer.",
            category=ToolCategory.RAG,
            function=tool_error_handler(answer_question),
            parameters={"question": "str - The question to answer"},
            returns={"answer": "str", "sources": "List of source documents used"}
        ))
        
        self.register(Tool(
            name="rag_stats",
            description="Get statistics about uploaded documents (count, chunks, etc.).",
            category=ToolCategory.RAG,
            function=tool_error_handler(get_document_stats),
            parameters={},
            returns={"total_documents": "int", "total_chunks": "int"}
        ))
        
        # ============== SQL TOOLS ==============
        
        self.register(Tool(
            name="sql_query",
            description="Execute a natural language query against the data. Converts to SQL and returns results.",
            category=ToolCategory.SQL,
            function=tool_error_handler(text_to_sql_agent.execute_query),
            parameters={"question": "str - Natural language question about the data"},
            returns={"sql": "str - Generated SQL", "results": "List of rows", "answer": "str - Natural language answer"},
            requires_data=True
        ))
        
        self.register(Tool(
            name="sql_schema",
            description="Get the schema of the uploaded data table. Shows available columns and types.",
            category=ToolCategory.SQL,
            function=tool_error_handler(text_to_sql_agent.get_schema_summary),
            parameters={},
            returns={"columns": "Dict of column names and types"},
            requires_data=True
        ))
        
        self.register(Tool(
            name="data_stats",
            description="Get high-level statistics about the uploaded data (total records, churn rate, etc.).",
            category=ToolCategory.SQL,
            function=tool_error_handler(get_churn_stats),
            parameters={},
            returns={"stats": "Dict with total_customers, churn_rate, avg_tenure, etc."},
            requires_data=True
        ))
        
        self.register(Tool(
            name="data_by_category",
            description="Get churn breakdown by categorical columns (contract type, payment method, etc.).",
            category=ToolCategory.SQL,
            function=tool_error_handler(get_churn_by_category),
            parameters={},
            returns={"by_contract": "List", "by_payment": "List", "etc.": "..."},
            requires_data=True
        ))
        
        # ============== ML TOOLS ==============
        
        self.register(Tool(
            name="ml_train",
            description="Train a machine learning model on the uploaded data. Auto-detects features.",
            category=ToolCategory.ML,
            function=tool_error_handler(churn_predictor.train),
            parameters={"algorithm": "str - 'random_forest' or 'gradient_boosting' (default: random_forest)"},
            returns={"accuracy": "float", "precision": "float", "recall": "float", "f1": "float", "feature_importance": "Dict"},
            requires_data=True
        ))
        
        @tool_error_handler
        def ml_predict_wrapper(customer_data: Dict[str, Any]) -> Dict[str, Any]:
            """Wrapper for predict that accepts dict"""
            return churn_predictor.predict(customer_data)
        
        self.register(Tool(
            name="ml_predict",
            description="Predict churn for a single customer. Requires trained model.",
            category=ToolCategory.ML,
            function=ml_predict_wrapper,
            parameters={"customer_data": "Dict - Customer features matching the training data columns"},
            returns={"prediction": "int (0/1)", "probability": "float", "risk_level": "str"},
            requires_model=True
        ))
        
        self.register(Tool(
            name="ml_batch_predict",
            description="Get predictions for multiple customers from the database.",
            category=ToolCategory.ML,
            function=tool_error_handler(churn_predictor.predict_batch),
            parameters={"limit": "int - Max customers to predict (default: 100)"},
            returns={"predictions": "List of predictions with probabilities"},
            requires_model=True
        ))
        
        @tool_error_handler
        def ml_feature_importance() -> Dict[str, Any]:
            """Get feature importance from trained model"""
            if not churn_predictor.is_trained:
                return {"success": False, "error": "Model not trained"}
            return {
                "success": True,
                "feature_importance": churn_predictor.feature_importance,
                "features_used": churn_predictor.training_stats.get("features_used", 0)
            }
        
        self.register(Tool(
            name="ml_feature_importance",
            description="Get feature importance scores from the trained model.",
            category=ToolCategory.ML,
            function=ml_feature_importance,
            parameters={},
            returns={"feature_importance": "Dict mapping feature names to importance scores"},
            requires_model=True
        ))
        
        @tool_error_handler
        def ml_model_status() -> Dict[str, Any]:
            """Check if model is trained and get its stats"""
            return {
                "is_trained": churn_predictor.is_trained,
                "algorithm": churn_predictor.training_stats.get("algorithm") if churn_predictor.is_trained else None,
                "accuracy": churn_predictor.training_stats.get("accuracy") if churn_predictor.is_trained else None,
                "features_used": churn_predictor.training_stats.get("features_used") if churn_predictor.is_trained else None
            }
        
        self.register(Tool(
            name="ml_model_status",
            description="Check if a ML model is trained and get its performance metrics.",
            category=ToolCategory.ML,
            function=ml_model_status,
            parameters={},
            returns={"is_trained": "bool", "algorithm": "str", "accuracy": "float"}
        ))
        
        # ============== VISUALIZATION TOOLS ==============
        
        self.register(Tool(
            name="chart_distribution",
            description="Generate a pie chart showing churn vs retention distribution.",
            category=ToolCategory.VISUALIZATION,
            function=tool_error_handler(generate_churn_distribution_chart),
            parameters={},
            returns={"chart": "base64 encoded PNG image"},
            requires_data=True
        ))
        
        self.register(Tool(
            name="chart_by_category",
            description="Generate bar charts showing churn rate by category (contract, payment, etc.).",
            category=ToolCategory.VISUALIZATION,
            function=tool_error_handler(generate_churn_by_category_chart),
            parameters={},
            returns={"chart": "base64 encoded PNG image"},
            requires_data=True
        ))
        
        self.register(Tool(
            name="chart_feature_importance",
            description="Generate a bar chart of ML model feature importance scores.",
            category=ToolCategory.VISUALIZATION,
            function=tool_error_handler(generate_feature_importance_chart),
            parameters={},
            returns={"chart": "base64 encoded PNG image"},
            requires_model=True
        ))
        
        self.register(Tool(
            name="chart_risk_distribution",
            description="Generate a histogram showing the distribution of churn risk scores.",
            category=ToolCategory.VISUALIZATION,
            function=tool_error_handler(generate_risk_distribution_chart),
            parameters={},
            returns={"chart": "base64 encoded PNG image"},
            requires_model=True
        ))
        
        self._initialized = True
        logger.info(f"Tool registry initialized with {len(self._tools)} tools")


# Global registry instance
tool_registry = ToolRegistry()


def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry, initializing if needed"""
    if not tool_registry._initialized:
        tool_registry.initialize_tools()
    return tool_registry
