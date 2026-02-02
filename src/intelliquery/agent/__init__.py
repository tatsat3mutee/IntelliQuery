"""
Agent Module - Planner-Based Agentic Architecture
=================================================
Implements autonomous goal-driven reasoning for IntelliQuery AI.

Components:
- base.py: Base classes, types, state management
- tools.py: Tool registry and definitions
- planner.py: LLM-powered planning agent
- executor.py: Stateful execution loop
- synthesizer.py: Insight synthesis agent
"""

from .base import AgentState, Tool, ExecutionStep, ExecutionPlan
from .tools import ToolRegistry, tool_registry
from .planner import PlannerAgent
from .executor import AgentExecutor
from .synthesizer import SynthesisAgent

__all__ = [
    'AgentState',
    'Tool', 
    'ExecutionStep',
    'ExecutionPlan',
    'ToolRegistry',
    'tool_registry',
    'PlannerAgent',
    'AgentExecutor',
    'SynthesisAgent'
]
