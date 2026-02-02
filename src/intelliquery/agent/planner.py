"""
Planner Agent - LLM-Powered Goal Decomposition
==============================================
Takes a user's natural language goal and decomposes it into an executable plan.

Features:
- Goal interpretation
- Multi-step plan generation
- Tool-aware reasoning
- Context-sensitive planning
"""

import logging
import json
import re
from typing import Dict, List, Any, Optional

from .base import ExecutionPlan, ExecutionStep, StepStatus
from .tools import get_tool_registry

logger = logging.getLogger(__name__)


class PlannerAgent:
    """
    LLM-powered agent that creates execution plans from user goals.
    
    The planner:
    1. Analyzes the user's goal
    2. Determines which tools are needed
    3. Orders operations logically
    4. Handles dependencies between steps
    """
    
    # Template for the planner prompt
    PLANNER_PROMPT = """You are an intelligent data analysis planner. Your job is to create an execution plan for the user's goal.

## Available Tools:
{tool_descriptions}

## Current Context:
- Data uploaded: {has_data}
- Documents uploaded: {has_documents}
- Model trained: {has_model}
- Available columns: {columns}

## User Goal:
{user_goal}

## Instructions:
1. Analyze what the user wants to achieve
2. Break it down into steps using ONLY the available tools
3. Order steps logically (data queries before analysis, training before prediction)
4. Return a JSON plan

## Output Format (return ONLY valid JSON):
{{
    "reasoning": "Brief explanation of your approach",
    "steps": [
        {{
            "step_id": 1,
            "tool_name": "tool_name_here",
            "description": "What this step accomplishes",
            "parameters": {{}},
            "depends_on": []
        }}
    ]
}}

## Rules:
- Use only tools from the Available Tools list
- If data is not uploaded, cannot use SQL or ML tools
- If model is not trained, cannot use prediction tools (but can train first)
- Keep plans focused - typically 2-5 steps
- Charts should come after the data they visualize

Create the execution plan:"""

    def __init__(self):
        self.tool_registry = None
        self._llm_client = None
    
    def _get_llm_client(self):
        """Get or create LLM client for planning"""
        if self._llm_client is None:
            from ..core.database import db_client
            self._llm_client = db_client
        return self._llm_client
    
    def _get_tool_registry(self):
        """Get tool registry (lazy load)"""
        if self.tool_registry is None:
            self.tool_registry = get_tool_registry()
        return self.tool_registry
    
    def _get_context(self) -> Dict[str, Any]:
        """Gather current context for planning"""
        context = {
            "has_data": False,
            "has_documents": False,
            "has_model": False,
            "columns": []
        }
        
        try:
            # Check for data
            from ..analytics.data_handler import get_churn_stats
            stats = get_churn_stats()
            if stats.get("success") and stats.get("stats", {}).get("total_customers", 0) > 0:
                context["has_data"] = True
            
            # Check for documents
            from ..rag.document_processor import get_document_stats
            doc_stats = get_document_stats()
            if doc_stats.get("success") and doc_stats.get("total_documents", 0) > 0:
                context["has_documents"] = True
            
            # Check for model
            from ..ml.predictor import churn_predictor
            context["has_model"] = churn_predictor.is_trained
            
            # Get columns
            from ..analytics.text_to_sql import text_to_sql_agent
            schema = text_to_sql_agent.get_schema_summary()
            if schema.get("columns"):
                context["columns"] = list(schema.get("columns", {}).keys())[:10]  # Top 10 columns
        except Exception as e:
            logger.warning(f"Error getting context: {e}")
        
        return context
    
    def _build_prompt(self, goal: str, context: Dict[str, Any]) -> str:
        """Build the planner prompt with context"""
        registry = self._get_tool_registry()
        
        return self.PLANNER_PROMPT.format(
            tool_descriptions=registry.get_tool_descriptions(),
            has_data="Yes" if context.get("has_data") else "No - user needs to upload data first",
            has_documents="Yes" if context.get("has_documents") else "No - user needs to upload documents first",
            has_model="Yes" if context.get("has_model") else "No - model needs to be trained first",
            columns=", ".join(context.get("columns", [])) if context.get("columns") else "None available",
            user_goal=goal
        )
    
    def _parse_llm_response(self, response: str) -> Dict[str, Any]:
        """Parse LLM response to extract JSON plan"""
        try:
            # Try direct JSON parse first
            return json.loads(response)
        except json.JSONDecodeError:
            pass
        
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```(?:json)?\s*([\s\S]*?)```', response)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass
        
        # Try to find JSON object in text
        json_match = re.search(r'\{[\s\S]*\}', response)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError:
                pass
        
        logger.error(f"Could not parse LLM response: {response[:500]}")
        return None
    
    def _create_fallback_plan(self, goal: str, context: Dict[str, Any]) -> ExecutionPlan:
        """Create a fallback plan using heuristics when LLM fails"""
        logger.info("Using fallback planning (heuristic-based)")
        
        goal_lower = goal.lower()
        steps = []
        step_id = 1
        
        # Analyze goal keywords to determine needed tools
        needs_data_query = any(word in goal_lower for word in [
            'how many', 'count', 'average', 'total', 'show', 'list', 
            'customers', 'records', 'data', 'statistics', 'stats'
        ])
        
        needs_ml = any(word in goal_lower for word in [
            'predict', 'train', 'model', 'churn', 'risk', 'forecast',
            'feature importance', 'classification'
        ])
        
        needs_chart = any(word in goal_lower for word in [
            'chart', 'graph', 'plot', 'visualize', 'visualization', 'show me'
        ])
        
        needs_documents = any(word in goal_lower for word in [
            'document', 'explain', 'what is', 'how does', 'describe',
            'according to', 'paper', 'article'
        ])
        
        # Build steps based on needs and available context
        if needs_documents and context.get("has_documents"):
            steps.append(ExecutionStep(
                step_id=step_id,
                tool_name="rag_answer",
                description="Search documents for relevant information",
                parameters={"question": goal}
            ))
            step_id += 1
        
        if needs_data_query and context.get("has_data"):
            steps.append(ExecutionStep(
                step_id=step_id,
                tool_name="sql_query",
                description="Query the data to answer the question",
                parameters={"question": goal}
            ))
            step_id += 1
        
        if needs_ml and context.get("has_data"):
            if not context.get("has_model"):
                steps.append(ExecutionStep(
                    step_id=step_id,
                    tool_name="ml_train",
                    description="Train a machine learning model",
                    parameters={"algorithm": "random_forest"}
                ))
                step_id += 1
            
            if "predict" in goal_lower:
                steps.append(ExecutionStep(
                    step_id=step_id,
                    tool_name="ml_batch_predict",
                    description="Generate predictions for customers",
                    parameters={"limit": 100},
                    depends_on=[step_id - 1] if not context.get("has_model") else []
                ))
                step_id += 1
            
            if "feature" in goal_lower or "importance" in goal_lower:
                steps.append(ExecutionStep(
                    step_id=step_id,
                    tool_name="ml_feature_importance",
                    description="Get feature importance from the model",
                    parameters={},
                    depends_on=[s.step_id for s in steps if s.tool_name == "ml_train"]
                ))
                step_id += 1
        
        if needs_chart and context.get("has_data"):
            if "distribution" in goal_lower or "churn" in goal_lower:
                steps.append(ExecutionStep(
                    step_id=step_id,
                    tool_name="chart_distribution",
                    description="Generate churn distribution chart",
                    parameters={}
                ))
                step_id += 1
            
            if "category" in goal_lower or "breakdown" in goal_lower:
                steps.append(ExecutionStep(
                    step_id=step_id,
                    tool_name="chart_by_category",
                    description="Generate category breakdown chart",
                    parameters={}
                ))
                step_id += 1
            
            if "feature" in goal_lower and context.get("has_model"):
                steps.append(ExecutionStep(
                    step_id=step_id,
                    tool_name="chart_feature_importance",
                    description="Generate feature importance chart",
                    parameters={}
                ))
                step_id += 1
        
        # Default: if no specific steps, get data stats
        if not steps and context.get("has_data"):
            steps.append(ExecutionStep(
                step_id=1,
                tool_name="data_stats",
                description="Get overall data statistics",
                parameters={}
            ))
        
        # If still no steps, inform user
        if not steps:
            steps.append(ExecutionStep(
                step_id=1,
                tool_name="data_stats",
                description="Check data availability",
                parameters={}
            ))
        
        return ExecutionPlan(
            goal=goal,
            steps=steps,
            reasoning="Fallback plan generated using keyword analysis",
            estimated_tools=[s.tool_name for s in steps]
        )
    
    def create_plan(self, goal: str) -> ExecutionPlan:
        """
        Create an execution plan for the given goal.
        
        Args:
            goal: Natural language description of what the user wants
            
        Returns:
            ExecutionPlan with ordered steps to achieve the goal
        """
        logger.info(f"Creating plan for goal: {goal}")
        
        # Get current context
        context = self._get_context()
        logger.info(f"Context: {context}")
        
        # Build prompt
        prompt = self._build_prompt(goal, context)
        
        # Try to use LLM for planning
        try:
            llm = self._get_llm_client()
            
            # Use the foundation model endpoint
            response = llm.chat_completion(
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1024
            )
            
            if response:
                plan_data = self._parse_llm_response(response)
                
                if plan_data and "steps" in plan_data:
                    # Convert to ExecutionPlan
                    steps = []
                    for step_data in plan_data["steps"]:
                        steps.append(ExecutionStep(
                            step_id=step_data.get("step_id", len(steps) + 1),
                            tool_name=step_data.get("tool_name", ""),
                            description=step_data.get("description", ""),
                            parameters=step_data.get("parameters", {}),
                            depends_on=step_data.get("depends_on", [])
                        ))
                    
                    plan = ExecutionPlan(
                        goal=goal,
                        steps=steps,
                        reasoning=plan_data.get("reasoning", ""),
                        estimated_tools=[s.tool_name for s in steps]
                    )
                    
                    logger.info(f"Created LLM plan with {len(steps)} steps")
                    return plan
        
        except Exception as e:
            logger.warning(f"LLM planning failed: {e}, using fallback")
        
        # Fallback to heuristic planning
        return self._create_fallback_plan(goal, context)
    
    def validate_plan(self, plan: ExecutionPlan) -> Dict[str, Any]:
        """
        Validate an execution plan.
        
        Checks:
        - All tool names are valid
        - Dependencies are valid
        - Required context is available
        """
        registry = self._get_tool_registry()
        context = self._get_context()
        
        errors = []
        warnings = []
        
        for step in plan.steps:
            # Check tool exists
            tool = registry.get(step.tool_name)
            if not tool:
                errors.append(f"Step {step.step_id}: Unknown tool '{step.tool_name}'")
                continue
            
            # Check dependencies
            step_ids = {s.step_id for s in plan.steps}
            for dep in step.depends_on:
                if dep not in step_ids:
                    errors.append(f"Step {step.step_id}: Invalid dependency on step {dep}")
            
            # Check requirements
            if tool.requires_data and not context.get("has_data"):
                warnings.append(f"Step {step.step_id}: Tool '{step.tool_name}' requires data, but none uploaded")
            
            if tool.requires_model and not context.get("has_model"):
                # Check if model training is in a previous step
                has_train_step = any(
                    s.tool_name == "ml_train" and s.step_id < step.step_id
                    for s in plan.steps
                )
                if not has_train_step:
                    warnings.append(f"Step {step.step_id}: Tool '{step.tool_name}' requires trained model")
        
        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings
        }


# Singleton instance
planner_agent = PlannerAgent()
