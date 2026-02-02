"""
IntelliQuery AI - Run Script

Usage:
    python run.py serve    # Start server
    python run.py test     # Test connection
"""

import sys
import uvicorn
from src.intelliquery.api.app import app
from src.intelliquery.core.config import config

def serve():
    """Start the FastAPI server"""
    print("Starting IntelliQuery AI Server...")
    print(f"Config: {config.get_status()}")
    uvicorn.run(
        "src.intelliquery.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

def test():
    """Test Databricks connection"""
    from src.intelliquery.core.database import db_client
    
    print("Testing Databricks connection...")
    if db_client.test_connection():
        print("[OK] Connection successful!")
    else:
        print("[ERROR] Connection failed!")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run.py [serve|test]")
        sys.exit(1)
    
    command = sys.argv[1]
    if command == "serve":
        serve()
    elif command == "test":
        test()
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
