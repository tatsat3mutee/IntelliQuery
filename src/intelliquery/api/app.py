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
"""

import logging
import asyncio
from pathlib import Path
from typing import Optional
from concurrent.futures import ThreadPoolExecutor

from fastapi import FastAPI, File, UploadFile, Form, Request, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles

from ..core.config import config
from ..core.database import db_client
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

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create app
app = FastAPI(
    title="IntelliQuery AI Simple",
    description="RAG + ML for Billing Data - Databricks Only",
    version="1.0.0"
)

# Setup templates
# Path: src/intelliquery/api/app.py -> go up 3 levels to project root, then to templates
TEMPLATE_DIR = Path(__file__).parent.parent.parent.parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

# Log template directory for debugging
logger.info(f"📂 Templates: {TEMPLATE_DIR}")

# Thread pool for blocking operations
if 'executor' not in dir():
    executor = ThreadPoolExecutor(max_workers=4)


# ============== WEB UI ==============

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render main UI"""
    return templates.TemplateResponse("index.html", {"request": request})


# ============== DOCUMENT UPLOAD (RAG) ==============

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload document for RAG embedding"""
    try:
        logger.info(f"=== Received upload request for: {file.filename} ===")
        
        # Validate file
        if not file.filename:
            return JSONResponse({"success": False, "error": "No filename"}, status_code=400)
        
        # Read content
        content = await file.read()
        logger.info(f"File size: {len(content)} bytes")
        
        # Extract text based on file type
        text = None
        filename_lower = file.filename.lower()
        
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
                logger.info(f"Extracted {len(text)} characters from {len(pdf_reader.pages)} PDF pages")
                
                if not text.strip():
                    return JSONResponse({
                        "success": False, 
                        "error": "PDF appears to be empty or contains only images"
                    }, status_code=400)
                    
            except ImportError:
                return JSONResponse({
                    "success": False, 
                    "error": "PDF support not installed. Install pypdf: pip install pypdf"
                }, status_code=500)
            except Exception as pdf_error:
                logger.error(f"PDF parsing error: {pdf_error}")
                return JSONResponse({
                    "success": False, 
                    "error": f"PDF parsing failed: {str(pdf_error)}"
                }, status_code=400)
        else:
            # Try to decode as text
            try:
                text = content.decode('utf-8')
            except UnicodeDecodeError:
                try:
                    text = content.decode('latin-1')
                except:
                    return JSONResponse({
                        "success": False, 
                        "error": "Cannot decode file. Please upload a text file (.txt) or PDF (.pdf)"
                    }, status_code=400)
        
        if not text or not text.strip():
            return JSONResponse({
                "success": False, 
                "error": "File is empty or contains no readable text"
            }, status_code=400)
        
        # Process document in background thread to avoid blocking
        logger.info(f"Starting document processing for {file.filename}")
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(executor, process_document, file.filename, text)
        logger.info(f"Processing complete: {result}")
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Upload document error: {e}", exc_info=True)
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============== CHURN DATA UPLOAD ==============

@app.post("/upload-churn")
async def upload_churn(file: UploadFile = File(...)):
    """Upload Telco Churn CSV/Excel data"""
    try:
        # Validate file
        if not file.filename:
            return JSONResponse({"success": False, "error": "No filename"}, status_code=400)
        
        # Check extension
        if not file.filename.endswith(('.csv', '.xlsx', '.xls')):
            return JSONResponse({
                "success": False, 
                "error": "Please upload a CSV or Excel file"
            }, status_code=400)
        
        # Read content
        content = await file.read()
        
        # Process Churn file
        result = process_churn_file(content, file.filename)
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Upload Churn error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


# ============== RAG QUESTION ANSWERING ==============

@app.post("/ask")
async def ask_question_endpoint(question: str = Form(...)):
    """Ask a question using RAG"""
    try:
        if not question.strip():
            return JSONResponse({"success": False, "error": "Empty question"}, status_code=400)
        
        result = answer_question(question)
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Ask error: {e}")
        return JSONResponse({"success": False, "error": str(e)}, status_code=500)


@app.post("/ask-intelligent")
async def ask_intelligent_endpoint(question: str = Form(...)):
    """
    Intelligent query handler - routes to appropriate backend:
    - KNOWLEDGE queries → Document RAG
    - DATA queries → Text-to-SQL
    - HYBRID queries → Both
    """
    try:
        if not question.strip():
            return JSONResponse({"success": False, "error": "Empty question"}, status_code=400)
        
        logger.info(f"Intelligent query: {question}")
        
        # Use the query router to classify and handle
        result = query_router.route_query(question)
        
        logger.info(f"Query type: {result.get('query_type')}, Success: {result.get('success')}")
        
        return JSONResponse(result)
    
    except Exception as e:
        logger.error(f"Intelligent ask error: {e}", exc_info=True)
        return JSONResponse({
            "success": False, 
            "error": str(e),
            "query_type": "error"
        }, status_code=500)


@app.post("/ask-data")
async def ask_data_endpoint(question: str = Form(...)):
    """Direct Text-to-SQL query endpoint"""
    try:
        if not question.strip():
            return JSONResponse({"success": False, "error": "Empty question"}, status_code=400)
        
        result = text_to_sql_agent.execute_query(question)
        return JSONResponse(result)
    
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
async def health_check():
    """Health check endpoint"""
    status = {
        "status": "healthy",
        "config": config.get_status(),
        "databricks_connection": False
    }
    
    # Test Databricks connection
    if config.is_configured():
        try:
            status["databricks_connection"] = db_client.test_connection()
        except Exception as e:
            status["databricks_error"] = str(e)
    
    return JSONResponse(status)


@app.get("/config")
async def get_config():
    """Get configuration status (no secrets)"""
    return JSONResponse(config.get_status())


# ============== VECTOR SEARCH MANAGEMENT ==============

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
