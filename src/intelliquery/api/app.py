"""
IntelliQuery AI Simple - FastAPI Application
========================================
2 Tables + ML Predictions + Graphs

Endpoints:
- /                  : Web UI
- /upload-document   : Upload docs for RAG
- /upload-churn      : Upload Telco Churn data
- /ask               : Ask questions (RAG)
- /train-model       : Train Churn ML model
- /predict-churn     : Get Churn prediction
- /get-charts        : Get all churn charts
- /health            : Health check

Enterprise Features:
- Rate limiting
- Input validation
- Audit logging
- Error handling framework
"""

import logging
import asyncio
import uuid
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ..core.config import config
from ..core.database import db_client
from ..core.security import InputValidator, FileType
from ..core.error_handler import (
    IntelliQueryException, ValidationException, FileValidationException,
    FileTooLargeException, UnsupportedFileTypeException, ErrorCode,
    get_request_id, safe_error_response
)
from ..core.middleware import (
    RequestContextMiddleware, RateLimitMiddleware, AuditLogMiddleware,
    timeout_decorator
)
from ..core.health import health_checker
from ..rag.document_processor import process_document, answer_question, get_document_stats
from ..rag.vector_search import vector_search_manager
from ..analytics.data_handler import process_churn_file, get_churn_stats, get_churn_by_category
from ..ml.predictor import churn_predictor
from ..analytics.query_router import query_router
from ..analytics.text_to_sql import text_to_sql_agent
from ..visualization.chart_generator import (
    generate_churn_distribution_chart,
    generate_churn_by_category_chart,
    generate_feature_importance_chart,
    generate_risk_distribution_chart,
    generate_prediction_comparison_chart
)
from ..agent.executor import agent_executor
from ..agent.synthesizer import synthesis_agent

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Setup audit logger
audit_logger = logging.getLogger("audit")
audit_logger.setLevel(logging.INFO)

# Create app
app = FastAPI(
    title="IntelliQuery AI Simple",
    description="RAG + ML for Billing Data - Databricks Only",
    version="1.0.0"
)

# Add enterprise middleware (order matters - first added = outermost)
app.add_middleware(AuditLogMiddleware, log_request_body=False)
app.add_middleware(RateLimitMiddleware)
app.add_middleware(RequestContextMiddleware)

# Setup templates
# Path: src/intelliquery/api/app.py -> go up 3 levels to project root, then to templates
TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Log template directory for debugging
logger.info(f"📂 Templates: {TEMPLATE_DIR}")

# Thread pool for blocking operations
if 'executor' not in dir():
    executor = ThreadPoolExecutor(max_workers=4)

# File upload limits
MAX_DOCUMENT_SIZE = 50 * 1024 * 1024  # 50MB
MAX_DATA_FILE_SIZE = 100 * 1024 * 1024  # 100MB
ALLOWED_DOCUMENT_TYPES = [FileType.PDF, FileType.TEXT]
ALLOWED_DATA_TYPES = [FileType.CSV, FileType.EXCEL_XLSX, FileType.EXCEL_XLS]


# ============== EXCEPTION HANDLERS ==============

@app.exception_handler(IntelliQueryException)
async def handle_intelliquery_exception(request: Request, exc: IntelliQueryException):
    """Handle custom IntelliQuery exceptions"""
    request_id = get_request_id(request)
    
    # Log with internal details
    log_msg = f"[{request_id}] {exc.error_code.value}: {exc.message}"
    if exc.internal_message:
        log_msg += f" | Internal: {exc.internal_message}"
    logger.warning(log_msg)
    
    return JSONResponse(
        status_code=exc.http_status,
        content=exc.to_response_dict(request_id)
    )


@app.exception_handler(Exception)
async def handle_generic_exception(request: Request, exc: Exception):
    """Handle unexpected exceptions - never expose internal details"""
    request_id = get_request_id(request)
    
    # Log full details internally
    logger.error(f"[{request_id}] Unhandled exception: {exc}", exc_info=True)
    
    # Return safe response
    return safe_error_response(
        error_code=ErrorCode.INTERNAL_ERROR,
        message="An unexpected error occurred. Please try again.",
        request_id=request_id,
        http_status=500
    )


# ============== WEB UI ==============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


# ============== DOCUMENT UPLOAD (RAG) ==============

@app.post("/upload-document")
async def upload_document(request: Request, file: UploadFile = File(...)):
    """Upload document for RAG embedding with enterprise-grade validation"""
    request_id = get_request_id(request)
    
    try:
        logger.info(f"[{request_id}] Document upload request: {file.filename}")
        
        # Validate filename
        if not file.filename:
            raise ValidationException("Filename is required", field="file")
        
        # Read content
        content = await file.read()
        
        # Validate file (size, type)
        validation = InputValidator.validate_file_upload(
            filename=file.filename,
            content=content,
            allowed_types=ALLOWED_DOCUMENT_TYPES,
            max_size_bytes=MAX_DOCUMENT_SIZE
        )
        
        if not validation.is_valid:
            raise FileValidationException(validation.error_message, file.filename)
        
        # Validate query input (filename can be used maliciously)
        filename = validation.sanitized_value
        logger.info(f"[{request_id}] Validated file: {len(content)} bytes")
        
        # Extract text based on file type
        text = None
        filename_lower = filename.lower()
        
        if filename_lower.endswith('.pdf'):
            # Parse PDF using pypdf (fast, direct)
            try:
                import io
                from pypdf import PdfReader
                
                pdf_file = io.BytesIO(content)
                pdf_reader = PdfReader(pdf_file)
                
                text_parts = []
                for page in pdf_reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text_parts.append(page_text)
                
                text = "\n\n".join(text_parts)
                logger.info(f"[{request_id}] Extracted {len(text)} chars from {len(pdf_reader.pages)} PDF pages")
                
                if not text.strip():
                    raise ValidationException(
                        "PDF appears to be empty or contains only images",
                        field="file"
                    )
                    
            except ImportError:
                raise ValidationException(
                    "PDF support not installed. Install pypdf: pip install pypdf",
                    field="file"
                )
            except ValidationException:
                raise
            except Exception as pdf_error:
                logger.error(f"[{request_id}] PDF parsing error: {pdf_error}")
                raise ValidationException(f"PDF parsing failed: {str(pdf_error)[:100]}", field="file")
        else:
            # Try to decode as text
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = content.decode('latin-1')
                except:
                    raise ValidationException(
                        "Cannot decode file. Please upload a text file (.txt) or PDF (.pdf)",
                        field="file"
                    )
        
        if not text or not text.strip():
            raise ValidationException(
                "File is empty or contains no readable text",
                field="file"
            )
        
        # Process document in background thread to avoid blocking
        logger.info(f"[{request_id}] Starting document processing")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_document, filename, text)
        logger.info(f"[{request_id}] Processing complete: {result.get('success')}")
        
        return JSONResponse(result)
    
    except IntelliQueryException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Upload document error: {e}", exc_info=True)
        raise


# ============== CHURN DATA UPLOAD ==============

@app.post("/upload-churn")
async def upload_churn(request: Request, file: UploadFile = File(...)):
    """Upload Telco Churn CSV/Excel data with validation"""
    request_id = get_request_id(request)
    
    try:
        # Validate filename
        if not file.filename:
            raise ValidationException("Filename is required", field="file")
        
        # Read content
        content = await file.read()
        
        # Validate file (size, type)
        validation = InputValidator.validate_file_upload(
            filename=file.filename,
            content=content,
            allowed_types=ALLOWED_DATA_TYPES,
            max_size_bytes=MAX_DATA_FILE_SIZE
        )
        
        if not validation.is_valid:
            raise FileValidationException(validation.error_message, file.filename)
        
        filename = validation.sanitized_value
        logger.info(f"[{request_id}] Processing churn file: {filename} ({len(content)} bytes)")
        
        # Process Churn file
        result = process_churn_file(content, filename)
        
        return JSONResponse(result)
    
    except IntelliQueryException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Upload Churn error: {e}")
        raise


# ============== RAG QUESTION ANSWERING ==============

@app.post("/ask")
async def ask_question_endpoint(request: Request, question: str = Form(...)):
    """Ask a question using RAG with input validation"""
    request_id = get_request_id(request)
    
    try:
        # Validate input
        validation = InputValidator.validate_query_input(question)
        if not validation.is_valid:
            raise ValidationException(validation.error_message, field="question")
        
        question = validation.sanitized_value
        logger.info(f"[{request_id}] RAG question: {question[:100]}...")
        
        result = answer_question(question)
        return JSONResponse(result)
    
    except IntelliQueryException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Ask error: {e}")
        raise


@app.post("/ask-intelligent")
async def ask_intelligent_endpoint(request: Request, question: str = Form(...)):
    """
    Intelligent query handler with validation - routes to appropriate backend:
    - KNOWLEDGE queries → Document RAG
    - DATA queries → Text-to-SQL
    - HYBRID queries → Both
    """
    request_id = get_request_id(request)
    
    try:
        # Validate input
        validation = InputValidator.validate_query_input(question)
        if not validation.is_valid:
            raise ValidationException(validation.error_message, field="question")
        
        question = validation.sanitized_value
        logger.info(f"[{request_id}] Intelligent query: {question[:100]}...")
        
        # Use the query router to classify and handle
        result = query_router.route_query(question)
        
        logger.info(f"[{request_id}] Query type: {result.get('query_type')}, Success: {result.get('success')}")
        
        return JSONResponse(result)
    
    except IntelliQueryException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Intelligent ask error: {e}", exc_info=True)
        raise


@app.post("/ask-data")
async def ask_data_endpoint(request: Request, question: str = Form(...)):
    """Direct Text-to-SQL query endpoint with validation"""
    request_id = get_request_id(request)
    
    try:
        # Validate input
        validation = InputValidator.validate_query_input(question)
        if not validation.is_valid:
            raise ValidationException(validation.error_message, field="question")
        
        question = validation.sanitized_value
        logger.info(f"[{request_id}] Data query: {question[:100]}...")
        
        result = text_to_sql_agent.execute_query(question)
        return JSONResponse(result)
    
    except IntelliQueryException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Data query error: {e}")
        raise
    
    except Exception as e:
        logger.error(f"Data query error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/data-schema")
async def get_data_schema():
    """Get the schema of the churn data table for Text-to-SQL"""
    try:
        schema = text_to_sql_agent.get_schema_summary()
        return JSONResponse({"success": True, **schema})
    except Exception as e:
        logger.error(f"Schema error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/example-queries")
async def get_example_queries():
    """Get example queries for each query type"""
    try:
        examples = query_router.get_example_queries()
        return JSONResponse({"success": True, "examples": examples})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============== ML MODEL ==============

@app.get("/train-model")
async def train_model(algorithm: str = "random_forest"):
    """Train Churn ML model on uploaded data"""
    try:
        result = churn_predictor.train(algorithm=algorithm)
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Train error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/predict-churn")
async def predict_churn(request: Request):
    """
    Get churn prediction for a customer - DATASET AGNOSTIC
    Accepts any form data matching the trained model's features
    """
    try:
        # Get form data dynamically
        form_data = await request.form()
        
        # Convert form data to proper types in a new dictionary
        customer_data: dict[str, int | float | str] = {}
        for key, value in form_data.items():
            if value and isinstance(value, str):
                # Try to convert to int
                if value.isdigit():
                    customer_data[key] = int(value)
                # Try to convert to float
                elif value.replace('.', '', 1).isdigit():
                    try:
                        customer_data[key] = float(value)
                    except:
                        customer_data[key] = value
                else:
                    customer_data[key] = value
            else:
                # Handle non-string values (though form data is typically strings)
                customer_data[key] = str(value) if value else ""
        
        logger.info(f"Prediction request with data: {customer_data}")
        result = churn_predictor.predict(customer_data)
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Predict error: {e}", exc_info=True)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/predict-batch")
async def predict_batch(limit: int = 100):
    """Get batch churn predictions"""
    try:
        result = churn_predictor.predict_batch(limit=limit)
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Batch predict error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============== CHARTS ==============

@app.get("/get-charts")
async def get_charts():
    """Get all churn charts as base64 images"""
    try:
        charts = {}
        
        # Generate all available charts
        churn_dist = generate_churn_distribution_chart()
        if churn_dist.get("success"):
            charts["churn_distribution"] = churn_dist.get("chart")
        
        category_chart = generate_churn_by_category_chart()
        if category_chart.get("success"):
            charts["churn_by_category"] = category_chart.get("chart")
        
        # Feature importance (if model is trained)
        if churn_predictor.is_trained:
            feature_chart = generate_feature_importance_chart()
            if feature_chart.get("success"):
                charts["feature_importance"] = feature_chart.get("chart")
            
            risk_chart = generate_risk_distribution_chart()
            if risk_chart.get("success"):
                charts["risk_distribution"] = risk_chart.get("chart")
        
        return JSONResponse({
            "success": True,
            "charts": charts,
            "model_trained": churn_predictor.is_trained
        })
    
    except Exception as e:
        logger.error(f"Charts error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/churn-stats")
async def get_churn_statistics():
    """Get churn data statistics"""
    try:
        stats = get_churn_stats()
        category_breakdown = get_churn_by_category()
        
        return JSONResponse({
            "success": True,
            "stats": stats.get("stats", {}),
            "by_category": category_breakdown.get("data", {})
        })
    
    except Exception as e:
        logger.error(f"Churn stats error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============== STATS & HEALTH ==============

@app.get("/stats")
async def get_stats():
    """Get system statistics"""
    try:
        doc_stats = get_document_stats()
        churn_stats = get_churn_stats()
        
        return JSONResponse({
            "success": True,
            "documents": doc_stats.get("stats", {}),
            "churn": churn_stats.get("stats", {}),
            "model": {
                "trained": churn_predictor.is_trained,
                "algorithm": churn_predictor.training_stats.get("algorithm") if churn_predictor.training_stats else None,
                "stats": churn_predictor.training_stats
            }
        })
    
    except Exception as e:
        logger.error(f"Stats endpoint error: {e}", exc_info=True)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/health")
async def health_check(deep: bool = False):
    """
    Comprehensive health check endpoint.
    
    Query params:
        deep: If true, includes slower checks (embedding/LLM services)
    """
    health = await health_checker.get_health(deep_check=deep)
    
    # Set appropriate status code based on health
    status_code = 200
    if health.status.value == "unhealthy":
        status_code = 503
    elif health.status.value == "degraded":
        status_code = 200  # Still operational, just degraded
    
    return JSONResponse(health.to_dict(), status_code=status_code)


@app.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe - is the process running?
    """
    return JSONResponse(health_checker.get_liveness())


@app.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe - is the service ready for traffic?
    """
    result = await health_checker.get_readiness()
    status_code = 200 if result["ready"] else 503
    return JSONResponse(result, status_code=status_code)


@app.get("/config")
async def get_config():
    """Get configuration status (no secrets)"""
    return JSONResponse(config.get_status())


# ============== VECTOR SEARCH MANAGEMENT ==============

@app.post("/ask-agentic")
async def ask_agentic_endpoint(request: Request, goal: str = Form(...)):
    """
    Planner-based agentic query handler with timeout protection.
    
    This is the main entry point for autonomous multi-step analysis.
    The agent will:
    1. Understand your goal
    2. Create an execution plan
    3. Execute tools (RAG, SQL, ML, Charts)
    4. Synthesize insights
    
    Example goals:
    - "Analyze customer churn and show me the key factors"
    - "Train a model and predict high-risk customers"
    - "Show me churn statistics and visualizations"
    """
    request_id = get_request_id(request)
    
    try:
        # Validate input
        validation = InputValidator.validate_query_input(goal)
        if not validation.is_valid:
            raise ValidationException(validation.error_message, field="goal")
        
        goal = validation.sanitized_value
        logger.info(f"[{request_id}] 🤖 Agentic query: {goal[:100]}...")
        
        # Run the agent with timeout protection
        try:
            state = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(
                    executor, agent_executor.run, goal
                ),
                timeout=120  # 2 minute timeout for complex queries
            )
        except asyncio.TimeoutError:
            from ..core.error_handler import TimeoutException
            raise TimeoutException(operation="Agent execution", timeout_seconds=120)
        
        # Synthesize results
        result = synthesis_agent.synthesize(state)
        
        # Add execution details
        result["execution"] = {
            "plan": state.plan.to_dict() if state.plan else None,
            "state_summary": state.get_summary(),
            "trace": agent_executor.get_execution_trace(state)
        }
        result["request_id"] = request_id
        
        logger.info(f"[{request_id}] Agent completed: {state.get_summary()}")
        
        return JSONResponse(result)
    
    except IntelliQueryException:
        raise
    except Exception as e:
        logger.error(f"[{request_id}] Agentic query error: {e}", exc_info=True)
        raise


@app.get("/agent/tools")
async def get_available_tools():
    """Get list of available tools for the agent"""
    try:
        from ..agent.tools import get_tool_registry
        registry = get_tool_registry()
        
        tools = []
        for tool in registry.list_tools():
            tools.append({
                "name": tool.name,
                "description": tool.description,
                "category": tool.category.value,
                "parameters": tool.parameters,
                "requires_data": tool.requires_data,
                "requires_model": tool.requires_model
            })
        
        return JSONResponse({
            "success": True,
            "tools": tools,
            "total": len(tools)
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/agent/plan")
async def create_agent_plan(goal: str = Form(...)):
    """Preview the execution plan without running it"""
    try:
        from ..agent.planner import planner_agent
        
        plan = planner_agent.create_plan(goal)
        validation = planner_agent.validate_plan(plan)
        
        return JSONResponse({
            "success": True,
            "plan": plan.to_dict(),
            "validation": validation
        })
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/vector-search/create-index")
async def create_vector_index(
    embedding_dimension: int = 384,
    distance_metric: str = "COSINE"
):
    """
    Create Databricks Vector Search index for semantic search
    
    This enables HIGH-PERFORMANCE search on millions of documents.
    Run this ONCE after uploading documents.
    """
    try:
        result = vector_search_manager.create_index(
            embedding_dimension=embedding_dimension,
            distance_metric=distance_metric
        )
        return JSONResponse(result)
    except Exception as e:
        logger.error(f"Create index error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/vector-search/sync")
async def sync_vector_index():
    """Manually trigger vector index sync (for TRIGGERED sync mode)"""
    try:
        result = vector_search_manager.sync_index()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.get("/vector-search/status")
async def vector_search_status():
    """Get vector search configuration and status"""
    try:
        status = {
            "enabled": config.USE_VECTOR_SEARCH_INDEX,
            "endpoint": config.VECTOR_SEARCH_ENDPOINT or "Not configured",
            "index_name": config.VECTOR_INDEX_NAME,
            "available": vector_search_manager.is_available(),
            "index_exists": vector_search_manager.index_exists()
        }
        
        if vector_search_manager.index_exists():
            index_info = vector_search_manager.get_index_info()
            status["index_info"] = index_info
        
        return JSONResponse({"success": True, **status})
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.delete("/vector-search/delete-index")
async def delete_vector_index():
    """Delete vector search index (WARNING: Cannot be undone!)"""
    try:
        result = vector_search_manager.delete_index()
        return JSONResponse(result)
    except Exception as e:
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============== STARTUP ==============

@app.on_event("startup")
async def startup():
    """Startup event"""
    logger.info("=" * 50)
    logger.info("🚀 IntelliQuery AI Simple - Starting")
    logger.info("=" * 50)
    logger.info(f"📂 Templates: {TEMPLATE_DIR}")
    logger.info(f"🗄️  Databricks configured: {config.is_configured()}")
    logger.info(f"📊 RAG Table: {config.RAG_TABLE}")
    logger.info(f"📈 Churn Table: {config.CHURN_TABLE}")
    logger.info("=" * 50)


@app.on_event("shutdown")
async def shutdown():
    """Cleanup on shutdown"""
    db_client.close()
    logger.info("👋 IntelliQuery AI Simple - Shutdown")


# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
