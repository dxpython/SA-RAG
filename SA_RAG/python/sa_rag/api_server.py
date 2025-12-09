import asyncio
import logging
import json
from typing import List, Optional, Dict, Any
from fastapi import FastAPI, HTTPException, BackgroundTasks, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from sa_rag.rag import RAGPipeline
from sa_rag.llm import LLMService
from sa_rag.embedding import EmbeddingService
from sa_rag.orchestrator import Orchestrator
from sa_rag.client import Client

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SA-RAG API", version="1.0.0")

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all for dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

from fastapi.exceptions import RequestValidationError
from fastapi.requests import Request
from fastapi.responses import JSONResponse

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.error(f"Validation error: {exc.errors()}")
    logger.error(f"Body: {await request.body()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": str(await request.body())},
    )

# Global RAG Client
rag_client: Optional[Client] = None

def get_client() -> Client:
    if not rag_client:
        raise HTTPException(status_code=503, detail="RAG system not initialized")
    return rag_client

@app.on_event("startup")
async def startup_event():
    global rag_client
    logger.info("Initializing RAG system...")
    
    # Initialize services via Client wrapper
    rag_client = Client(
        llm_provider="mock",
        embedding_provider="mock"
    )
    
    logger.info("RAG system initialized.")

# --- Models ---

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5
    use_graph: bool = True
    use_memory: bool = True
    filters: Optional[Dict[str, Any]] = None

class CreateDocumentRequest(BaseModel):
    title: str
    content: str
    generate_embeddings: bool = True

class MemoryRequest(BaseModel):
    content: str
    importance: float = 0.5

# --- Endpoints ---

@app.get("/healthz")
async def health_check():
    return {"status": "ok"}

@app.post("/api/v1/rag/query")
async def query(request: QueryRequest):
    client = get_client()
    try:
        response = client.ask(
            question=request.query
            # Using defaults for now as client.ask signature might simplify args
        )
        return response
    except Exception as e:
        logger.error(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rag/debug")
async def debug_query(request: QueryRequest):
    client = get_client()
    try:
        # Assuming r.debug_retrieval(query, top_k, ...) exists on pipeline
        debug_info = client.pipeline.debug_retrieval(
            query=request.query,
            top_k=request.top_k
        )
        return debug_info
    except Exception as e:
        logger.error(f"Debug query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rag/stream")
async def stream_query(request: QueryRequest):
    # Mock streaming for now since main pipeline might not support async generator yet
    # Or implement a simple simulation
    async def event_generator():
        client = get_client()
        
        # Simulate analysis
        yield f"data: {json.dumps({'type': 'analysis', 'data': {'intent': 'informational'}})}\n\n"
        await asyncio.sleep(0.5)
        
        # Simulate retrieval
        yield f"data: {json.dumps({'type': 'retrieval', 'data': {'nodes': 3}})}\n\n"
        await asyncio.sleep(0.5)
        
        # Get full answer
        answer = client.ask(request.query)
        full_text = answer.get("answer", "")
        
        # Stream tokens
        chunk_size = 5
        for i in range(0, len(full_text), chunk_size):
            chunk = full_text[i:i+chunk_size]
            yield f"data: {json.dumps({'type': 'answer', 'data': {'text': chunk}})}\n\n"
            await asyncio.sleep(0.1)
            
    return StreamingResponse(event_generator(), media_type="text/event-stream")

@app.get("/api/v1/documents")
async def list_documents(page: int = 1, page_size: int = 20):
    # This requires persistence layer interaction
    # For now, we return empty or what's in memory if Engine supports listing
    # The RustCoreEngine might expose doc ids
    # We will mock a response if persistence isn't fully ready
    return {
        "documents": [],
        "total": 0
    }

from sa_rag.file_parser import FileParser

# ...

@app.post("/api/v1/documents")
async def create_document(
    request: Request,
    file: Optional[UploadFile] = File(None),
    title: Optional[str] = Form(None),
    content: Optional[str] = Form(None),
    generate_embeddings: bool = Form(True)
):
    client = get_client()
    try:
        # Debug logging
        form_data = await request.form()
        logger.info(f"Form keys received: {list(form_data.keys())}")
        if file:
             logger.info(f"File received: {file.filename}, Content-Type: {file.content_type}")
        else:
             logger.info("No file received in 'file' parameter")

        doc_content = ""
        doc_title = title or "Untitled Document"
        
        # Priority: File > Content
        if file:
            logger.info(f"Processing file: {file.filename}")
            content_bytes = await file.read()
            doc_content = FileParser.parse(file.filename, content_bytes)
            doc_title = title or file.filename
        elif content:
            doc_content = content
        else:
             raise HTTPException(status_code=400, detail="Either file or content must be provided")

        doc_id = client.index(doc_content, generate_embeddings=generate_embeddings)
        
        # Construct a mock document object to return
        doc = {
            "docId": doc_id[0] if doc_id else 1,
            "title": doc_title,
            "content": doc_content, # Warning: This might be large
            "status": "ready",
            "nodeCount": 0, # Need to fetch
            "version": 1,
            "createdAt": "2023-01-01T00:00:00Z",
            "updatedAt": "2023-01-01T00:00:00Z"
        }
        return {"document": doc}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Index error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.delete("/api/v1/documents/{doc_id}")
async def delete_document(doc_id: int):
    # Not implemented in core yet
    return {"success": True}

@app.get("/api/v1/documents/{doc_id}/nodes")
async def get_document_nodes(doc_id: int, page: int = 1, page_size: int = 20):
    client = get_client()
    # Assuming client has a method to get nodes
    # If not, we might need to add one or mock
    # rust_core.get_node_info(node_id) exists
    return {
        "nodes": [],
        "total": 0
    }

@app.get("/api/v1/documents/{doc_id}/graph")
async def get_graph(doc_id: int):
    client = get_client()
    try:
        stats = client.pipeline.engine.get_graph_stats()
        # We need a proper graph export. 
        # For now, return a mock structure compatible with frontend
        return {
            "graph": {
                "nodes": [],
                "edges": [],
                "statistics": stats
            }
        }
    except Exception as e:
        logger.error(f"Graph error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/v1/rag/memory")
async def add_memory(request: MemoryRequest):
    client = get_client()
    try:
        client.add_memory(request.content)
        return {
            "memory": {
                "id": 1,
                "content": request.content,
                "importance": request.importance,
                "timestamp": "now"
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
