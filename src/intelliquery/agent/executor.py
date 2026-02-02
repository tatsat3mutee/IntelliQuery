"""
Agent Executor - Stateful Plan Execution Engine
===============================================
Executes plans created by the Planner Agent, managing state and handling errors.

Features:
- Sequential step execution
- Dependency resolution
- State management
- Error recovery
- Progress tracking
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

from .base import AgentState, ExecutionPlan, ExecutionStep, StepStatus
from .tools import get_tool_registry
from .planner import PlannerAgent, planner_agent

logger = logging.getLogger(__name__)


class AgentExecutor:
    """
    Executes plans and manages agent state.
    
    The executor:
    1. Takes a plan from the Planner
    2. Executes steps in order, respecting dependencies
    3. Accumulates results in AgentState
    4. Handles errors gracefully
    5. Provides execution trace for explainability
    """
    
    def __init__(self, planner: Optional[PlannerAgent] = None):
        self.planner = planner or planner_agent
        self.tool_registry = None
        self.current_state: Optional[AgentState] = None
    
    def _get_tool_registry(self):
        """Get tool registry (lazy load)"""
        if self.tool_registry is None:
            self.tool_registry = get_tool_registry()
        return self.tool_registry
    
    def _execute_step(self, step: ExecutionStep, state: AgentState) -> Dict[str, Any]:
        """Execute a single step and update state"""
        logger.info(f"Executing step {step.step_id}: {step.tool_name}")
        
        step.status = StepStatus.RUNNING
        step.started_at = datetime.now()
        
        registry = self._get_tool_registry()
        tool = registry.get(step.tool_name)
        
        if not tool:
            step.status = StepStatus.FAILED
            step.error = f"Tool '{step.tool_name}' not found"
            step.completed_at = datetime.now()
            return {"success": False, "error": step.error}
        
        try:
            # Execute the tool with parameters
            result = tool.execute(**step.parameters)
            
            if result.get("success"):
                step.status = StepStatus.COMPLETED
                step.result = result
                
                # Update state with result
                state.update_from_tool_result(step.tool_name, result)
                
                logger.info(f"Step {step.step_id} completed successfully")
            else:
                step.status = StepStatus.FAILED
                step.error = result.get("error", "Unknown error")
                state.errors.append(f"Step {step.step_id} ({step.tool_name}): {step.error}")
                
                logger.warning(f"Step {step.step_id} failed: {step.error}")
            
            step.completed_at = datetime.now()
            return result
            
        except Exception as e:
            step.status = StepStatus.FAILED
            step.error = str(e)
            step.completed_at = datetime.now()
            state.errors.append(f"Step {step.step_id} ({step.tool_name}): {str(e)}")
            
            logger.error(f"Step {step.step_id} exception: {e}")
            return {"success": False, "error": str(e)}
    
    def _can_execute_step(self, step: ExecutionStep, plan: ExecutionPlan) -> bool:
        """Check if a step can be executed (dependencies met)"""
        if step.status != StepStatus.PENDING:
            return False
        
        # Check all dependencies are completed
        for dep_id in step.depends_on:
            dep_step = next((s for s in plan.steps if s.step_id == dep_id), None)
            if dep_step and dep_step.status != StepStatus.COMPLETED:
                return False
        
        return True
    
    def execute_plan(self, plan: ExecutionPlan, state: Optional[AgentState] = None) -> AgentState:
        """
        Execute an entire plan.
        
        Args:
            plan: The execution plan to run
            state: Optional existing state to continue from
            
        Returns:
            Final AgentState with all results
        """
        # Initialize or use existing state
        if state is None:
            state = AgentState(goal=plan.goal, plan=plan)
        else:
            state.plan = plan
        
        self.current_state = state
        
        logger.info(f"Starting plan execution: {len(plan.steps)} steps")
        
        # Execute steps in order
        max_iterations = len(plan.steps) * 2  # Safety limit
        iteration = 0
        
        while not plan.is_complete() and iteration < max_iterations:
            iteration += 1
            
            # Find next executable step
            executable = [s for s in plan.steps if self._can_execute_step(s, plan)]
            
            if not executable:
                # No steps can execute - check for blocked steps
                pending = [s for s in plan.steps if s.status == StepStatus.PENDING]
                if pending:
                    # Steps are blocked by failed dependencies
                    for step in pending:
                        step.status = StepStatus.SKIPPED
                        step.error = "Skipped due to failed dependency"
                break
            
            # Execute first available step
            step = executable[0]
            self._execute_step(step, state)
        
        # Mark completion
        state.completed_at = datetime.now()
        
        # Log summary
        progress = plan.get_progress()
        logger.info(f"Plan execution complete: {progress}")
        
        return state
    
    def run(self, goal: str) -> AgentState:
        """
        Complete agent run: plan and execute.
        
        This is the main entry point for running the agent.
        
        Args:
            goal: Natural language goal from the user
            
        Returns:
            AgentState with all results
        """
        logger.info(f"Agent run started for goal: {goal}")
        
        # Create state
        state = AgentState(goal=goal)
        
        # Create plan
        try:
            plan = self.planner.create_plan(goal)
            state.plan = plan
            
            # Validate plan
            validation = self.planner.validate_plan(plan)
            if validation.get("warnings"):
                for warning in validation["warnings"]:
                    logger.warning(f"Plan warning: {warning}")
            
            if not validation.get("valid"):
                for error in validation.get("errors", []):
                    state.errors.append(f"Plan validation: {error}")
                state.completed_at = datetime.now()
                return state
            
            logger.info(f"Plan created with {len(plan.steps)} steps: {[s.tool_name for s in plan.steps]}")
            
        except Exception as e:
            state.errors.append(f"Planning failed: {str(e)}")
            state.completed_at = datetime.now()
            logger.error(f"Planning failed: {e}")
            return state
        
        # Execute plan
        return self.execute_plan(plan, state)
    
    def get_execution_trace(self, state: AgentState) -> List[Dict[str, Any]]:
        """Get detailed execution trace for debugging/explainability"""
        if not state.plan:
            return []
        
        trace = []
        for step in state.plan.steps:
            trace.append({
                "step_id": step.step_id,
                "tool": step.tool_name,
                "description": step.description,
                "status": step.status.value,
                "duration_ms": (
                    (step.completed_at - step.started_at).total_seconds() * 1000
                    if step.started_at and step.completed_at else None
                ),
                "error": step.error
            })
        
        return trace


# Singleton instance
agent_executor = AgentExecutor()
