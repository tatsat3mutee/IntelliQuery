# 🤖 IntelliQuery AI - Planner-Based Agentic Architecture

## Executive Summary

This document outlines the transformation of IntelliQuery AI from a **router-based reactive system** to a **planner-based autonomous agent** capable of goal-driven multi-step reasoning.

---

## ✅ Feasibility Assessment

### Current State Analysis

| Component | Current Implementation | Agentic Ready? |
|-----------|----------------------|----------------|
| **Document RAG** | ✅ Fully implemented (vector search, embeddings) | ✅ Ready as tool |
| **Text-to-SQL** | ✅ Dynamic schema detection, NL to SQL | ✅ Ready as tool |
| **ML Predictor** | ✅ Dataset-agnostic, auto-detection | ✅ Ready as tool |
| **Chart Generator** | ✅ Multiple chart types | ✅ Ready as tool |
| **Query Router** | ⚠️ Keyword-based classification | ❌ Replace with Planner |
| **FastAPI Layer** | ✅ RESTful endpoints | ⚠️ Add agentic endpoint |

### Feasibility: **HIGH** ✅

**Rationale:**
1. All core tools already exist and work independently
2. Modular design allows easy tool abstraction
3. No architectural blockers
4. Databricks integration already handles data layer

---

## 🏗️ Architecture Transformation

### Before (Router-Based)
```
User Query → Query Router → [RAG | SQL | ML] → Response
                 ↓
         Static keyword matching
         Single-step execution
         No state management
```

### After (Planner-Based)
```
User Goal → Planner Agent → Execution Plan → Tool Execution Loop → Synthesis → Response
                 ↓                  ↓                  ↓
         LLM goal understanding    Multi-step      State tracking
         Tool-aware planning       Autonomous      Iterative refinement
```

---

## 📋 Implementation Plan

### Phase 1: Tool Registry (2-3 hours)
Create a standardized tool interface that wraps existing modules.

**Files to create:**
- `src/intelliquery/agent/tools.py` - Tool definitions & registry
- `src/intelliquery/agent/base.py` - Base classes

**Tools to register:**
| Tool Name | Wraps | Input | Output |
|-----------|-------|-------|--------|
| `rag_search` | `document_processor.search_documents` | query: str | chunks: List[Dict] |
| `rag_answer` | `document_processor.answer_question` | query: str | answer: str, sources: List |
| `sql_query` | `text_to_sql.execute_query` | question: str | results: List, sql: str |
| `sql_schema` | `text_to_sql.get_schema_summary` | - | columns: Dict |
| `ml_train` | `predictor.train` | algorithm: str | metrics: Dict |
| `ml_predict` | `predictor.predict` | data: Dict | prediction: Dict |
| `ml_batch_predict` | `predictor.predict_batch` | limit: int | predictions: List |
| `chart_distribution` | `chart_generator.generate_churn_distribution_chart` | - | chart: base64 |
| `chart_category` | `chart_generator.generate_churn_by_category_chart` | - | chart: base64 |
| `chart_features` | `chart_generator.generate_feature_importance_chart` | - | chart: base64 |

---

### Phase 2: Planner Agent (3-4 hours)
Create the LLM-powered planner that decomposes goals into executable steps.

**File:** `src/intelliquery/agent/planner.py`

**Features:**
- Goal interpretation
- Multi-step plan generation
- Tool-aware reasoning
- Plan validation

**Planner Prompt Template:**
```
You are a data analysis planner. Given a user goal, create an execution plan.

Available Tools:
{tool_descriptions}

User Goal: {user_goal}
Data Context: {available_data}

Output a JSON execution plan with ordered steps.
```

---

### Phase 3: Execution Loop (3-4 hours)
Create the stateful execution engine that runs plans.

**File:** `src/intelliquery/agent/executor.py`

**Features:**
- State management
- Tool invocation
- Error handling & recovery
- Intermediate result tracking
- Satisfaction checking

**State Schema:**
```python
AgentState = {
    "goal": str,
    "plan": List[Step],
    "current_step": int,
    "documents": Optional[List],
    "sql_results": Optional[Dict],
    "model_metrics": Optional[Dict],
    "predictions": Optional[List],
    "charts": List[str],
    "insights": List[str],
    "errors": List[str],
    "completed": bool
}
```

---

### Phase 4: Synthesis Agent (2-3 hours)
Create the agent that transforms execution results into human insights.

**File:** `src/intelliquery/agent/synthesizer.py`

**Features:**
- Natural language explanation generation
- Evidence-backed insights
- Recommendation generation
- Confidence scoring

---

### Phase 5: API Integration (2 hours)
Add agentic endpoint to FastAPI.

**Endpoint:** `POST /ask-agentic`

**Request:**
```json
{
    "goal": "Explain why churn increased and recommend actions"
}
```

**Response:**
```json
{
    "success": true,
    "goal": "...",
    "insights": "...",
    "charts": [...],
    "recommendations": [...],
    "execution_trace": [...]
}
```

---

## 📁 New File Structure

```
src/intelliquery/
├── agent/                    # NEW: Agentic components
│   ├── __init__.py
│   ├── base.py              # Base classes, types, state
│   ├── tools.py             # Tool registry & definitions
│   ├── planner.py           # LLM planner agent
│   ├── executor.py          # Execution loop
│   └── synthesizer.py       # Insight synthesis
├── analytics/               # EXISTING (unchanged)
├── api/                     # EXISTING (add endpoint)
├── core/                    # EXISTING (unchanged)
├── ml/                      # EXISTING (unchanged)
├── rag/                     # EXISTING (unchanged)
└── visualization/           # EXISTING (unchanged)
```

---

## ⏱️ Timeline Estimate

| Phase | Component | Effort | Priority |
|-------|-----------|--------|----------|
| 1 | Tool Registry | 2-3 hours | P0 |
| 2 | Planner Agent | 3-4 hours | P0 |
| 3 | Execution Loop | 3-4 hours | P0 |
| 4 | Synthesis Agent | 2-3 hours | P1 |
| 5 | API Integration | 2 hours | P0 |
| **Total** | | **12-16 hours** | |

---

## 🎯 Success Criteria

1. **Multi-step queries work**: "Analyze churn, train a model, show feature importance"
2. **Goal-driven behavior**: System determines HOW to answer, not just WHAT to route to
3. **State persistence**: Intermediate results accessible across steps
4. **Explainability**: Users can see execution trace
5. **Backward compatibility**: Existing endpoints still work

---

## 🔮 Future Enhancements (Post-MVP)

1. **LangGraph Integration**: Structured agent orchestration
2. **Multi-Agent Collaboration**: Specialized agents for different tasks
3. **Streaming**: Real-time execution status
4. **Memory**: Cross-session learning
5. **Domain Planners**: Industry-specific reasoning templates

---

## ✅ Recommendation

**PROCEED WITH IMPLEMENTATION**

The current codebase provides all necessary building blocks. The transformation is:
- Technically feasible
- Incrementally deployable
- Backward compatible
- High value-add for users

Start with Phase 1 (Tool Registry) and iterate.
