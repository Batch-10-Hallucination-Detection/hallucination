"""
Web Platform and API Server Launcher.
Starts the FastAPI application server serving the interactive frontend and REST API at:
http://127.0.0.1:8000/
"""

import uvicorn
from backend.database.session import init_db

if __name__ == "__main__":
    print("=" * 70)
    print("Starting Hallucination Detection Web Research Platform...")
    print("Interactive Web Frontend: http://127.0.0.1:8000/")
    print("FastAPI Interactive Docs: http://127.0.0.1:8000/docs")
    print("=" * 70)
    init_db()
    uvicorn.run("backend.app.main:app", host="127.0.0.1", port=8000, reload=False)

