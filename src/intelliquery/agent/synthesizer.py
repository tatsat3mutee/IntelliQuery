"""
Synthesis Agent - Transform Results into Insights
=================================================
Takes execution state and produces human-readable insights, explanations, and recommendations.

Features:
- Natural language summary generation
- Evidence-backed insights
- Actionable recommendations
- Confidence scoring
"""

import logging
from typing import Dict, List, Any, Optional

from .base import AgentState

logger = logging.getLogger(__name__)


class SynthesisAgent:
    """
    Synthesizes agent execution results into user-friendly insights.
    
    The synthesizer:
    1. Analyzes accumulated state (SQL results, predictions, charts)
    2. Generates natural language explanations
    3. Provides evidence-backed recommendations
    4. Formats output for display
    """
    
    SYNTHESIS_PROMPT = """You are a data analyst presenting findings. Based on the following analysis results, provide a clear summary with insights and recommendations.

## User's Goal:
{goal}

## Analysis Results:

### Data Query Results:
{sql_results}

### ML Model Insights:
{model_insights}

### Key Statistics:
{statistics}

## Instructions:
1. Summarize what was found
2. Highlight key insights with specific numbers
3. Provide 2-3 actionable recommendations
4. Keep it concise (3-4 paragraphs max)

Write the summary:"""

    def __init__(self):
        self._llm_client = None
    
    def _get_llm_client(self):
        """Get LLM client for synthesis"""
        if self._llm_client is None:
            from ..core.database import db_client
            self._llm_client = db_client
        return self._llm_client
    
    def _format_sql_results(self, state: AgentState) -> str:
        """Format SQL results for the prompt"""
        if not state.sql_results:
            return "No data queries were executed."
        
        formatted = []
        for i, result in enumerate(state.sql_results, 1):
            sql = result.get("sql", "N/A")
            rows = result.get("results", [])
            row_count = result.get("row_count", len(rows))
            
            formatted.append(f"Query {i}: {sql}")
            formatted.append(f"  Results: {row_count} rows")
            
            # Show first few results
            if rows and len(rows) > 0:
                formatted.append(f"  Sample: {rows[:3]}")
        
        return "\n".join(formatted)
    
    def _format_model_insights(self, state: AgentState) -> str:
        """Format ML model insights for the prompt"""
        insights = []
        
        if state.model_metrics:
            metrics = state.model_metrics
            insights.append(f"Model trained: {metrics.get('algorithm', 'Unknown')}")
            insights.append(f"Accuracy: {metrics.get('accuracy', 'N/A')}")
            insights.append(f"Precision: {metrics.get('precision', 'N/A')}")
            insights.append(f"Recall: {metrics.get('recall', 'N/A')}")
            
            # Feature importance
            feat_imp = metrics.get("feature_importance", {})
            if feat_imp:
                top_features = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:5]
                insights.append("Top features: " + ", ".join([f"{f[0]} ({f[1]:.3f})" for f in top_features]))
        
        if state.predictions:
            insights.append(f"Predictions generated: {len(state.predictions)}")
            
            # Calculate risk distribution
            high_risk = sum(1 for p in state.predictions if p.get("probability", 0) > 0.7)
            medium_risk = sum(1 for p in state.predictions if 0.3 <= p.get("probability", 0) <= 0.7)
            low_risk = sum(1 for p in state.predictions if p.get("probability", 0) < 0.3)
            
            if high_risk or medium_risk or low_risk:
                insights.append(f"Risk distribution: High={high_risk}, Medium={medium_risk}, Low={low_risk}")
        
        return "\n".join(insights) if insights else "No ML analysis was performed."
    
    def _format_statistics(self, state: AgentState) -> str:
        """Extract and format statistics from state"""
        stats = []
        
        # From SQL results
        for result in state.sql_results:
            if result.get("results") and len(result["results"]) == 1:
                # Single row result - likely aggregation
                row = result["results"][0]
                for key, value in row.items():
                    if isinstance(value, (int, float)):
                        stats.append(f"{key}: {value}")
        
        # From context
        context_stats = state.context.get("stats", {})
        for key, value in context_stats.items():
            stats.append(f"{key}: {value}")
        
        return "\n".join(stats) if stats else "No specific statistics available."
    
    def synthesize(self, state: AgentState) -> Dict[str, Any]:
        """
        Synthesize execution state into insights.
        
        Args:
            state: AgentState with all execution results
            
        Returns:
            Dict with summary, insights, recommendations, and charts
        """
        logger.info("Synthesizing results...")
        
        # Collect existing insights from state
        collected_insights = state.insights.copy()
        
        # Try LLM-based synthesis
        try:
            llm = self._get_llm_client()
            
            prompt = self.SYNTHESIS_PROMPT.format(
                goal=state.goal,
                sql_results=self._format_sql_results(state),
                model_insights=self._format_model_insights(state),
                statistics=self._format_statistics(state)
            )
            
            response = llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=512
            )
            
            if response:
                collected_insights.insert(0, response)
                logger.info("LLM synthesis successful")
        
        except Exception as e:
            logger.warning(f"LLM synthesis failed: {e}, using fallback")
        
        # Fallback: Generate summary from state
        summary = self._generate_fallback_summary(state)
        recommendations = self._generate_recommendations(state)
        
        return {
            "success": True,
            "goal": state.goal,
            "summary": collected_insights[0] if collected_insights else summary,
            "insights": collected_insights,
            "recommendations": recommendations,
            "charts": state.charts,
            "has_errors": len(state.errors) > 0,
            "errors": state.errors,
            "execution_trace": self._get_execution_summary(state)
        }
    
    def _generate_fallback_summary(self, state: AgentState) -> str:
        """Generate a summary without LLM"""
        parts = []
        
        parts.append(f"**Analysis for:** {state.goal}")
        parts.append("")
        
        # Document results
        if state.documents:
            parts.append(f"📄 Found {len(state.documents)} relevant document sections.")
        
        # SQL results
        if state.sql_results:
            total_rows = sum(r.get("row_count", 0) for r in state.sql_results)
            parts.append(f"📊 Executed {len(state.sql_results)} data queries, returning {total_rows} total rows.")
        
        # Model results
        if state.model_metrics:
            accuracy = state.model_metrics.get("accuracy", "N/A")
            parts.append(f"🤖 ML model trained with {accuracy} accuracy.")
        
        # Predictions
        if state.predictions:
            parts.append(f"🎯 Generated {len(state.predictions)} predictions.")
        
        # Charts
        if state.charts:
            parts.append(f"📈 Created {len(state.charts)} visualizations.")
        
        # Errors
        if state.errors:
            parts.append(f"⚠️ {len(state.errors)} issues encountered during analysis.")
        
        return "\n".join(parts)
    
    def _generate_recommendations(self, state: AgentState) -> List[str]:
        """Generate recommendations based on state"""
        recommendations = []
        
        # Based on model metrics
        if state.model_metrics:
            feat_imp = state.model_metrics.get("feature_importance", {})
            if feat_imp:
                top_feature = max(feat_imp.items(), key=lambda x: x[1])[0]
                recommendations.append(
                    f"Focus on '{top_feature}' - it's the strongest predictor in your model."
                )
            
            accuracy = state.model_metrics.get("accuracy", 0)
            if isinstance(accuracy, (int, float)) and accuracy < 0.8:
                recommendations.append(
                    "Consider adding more features or collecting more data to improve model accuracy."
                )
        
        # Based on predictions
        if state.predictions:
            high_risk_count = sum(1 for p in state.predictions if p.get("probability", 0) > 0.7)
            total = len(state.predictions)
            
            if high_risk_count > 0:
                percentage = (high_risk_count / total) * 100
                recommendations.append(
                    f"{high_risk_count} customers ({percentage:.1f}%) are at high churn risk. "
                    "Prioritize retention efforts for these customers."
                )
        
        # Based on errors
        if state.errors:
            if any("data" in e.lower() for e in state.errors):
                recommendations.append(
                    "Upload your dataset to enable data analysis and ML predictions."
                )
            if any("model" in e.lower() for e in state.errors):
                recommendations.append(
                    "Train a model first to enable predictions and feature analysis."
                )
        
        # Default recommendation
        if not recommendations:
            recommendations.append(
                "Explore more queries to gain deeper insights into your data."
            )
        
        return recommendations
    
    def _get_execution_summary(self, state: AgentState) -> Dict[str, Any]:
        """Get a summary of the execution for transparency"""
        if not state.plan:
            return {"steps_executed": 0}
        
        progress = state.plan.get_progress()
        
        return {
            "total_steps": len(state.plan.steps),
            "completed": progress.get("completed", 0),
            "failed": progress.get("failed", 0),
            "skipped": progress.get("skipped", 0),
            "tools_used": state.plan.estimated_tools,
            "reasoning": state.plan.reasoning
        }


# Singleton instance
synthesis_agent = SynthesisAgent()
