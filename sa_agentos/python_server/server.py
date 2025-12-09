"""
SA-AgentOS Python Server
Provides HTTP API for SA-RAG Core to be consumed by C# Agent
"""

import os
import sys
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn

# Add parent directory to path to import sa_rag
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'SA_RAG', 'python'))

try:
    from sa_rag import RAG
    SA_RAG_AVAILABLE = True
except ImportError as e:
    SA_RAG_AVAILABLE = False
    print(f"Warning: sa_rag not available: {e}")

app = FastAPI(
    title="SA-AgentOS RAG API",
    description="HTTP API for SA-RAG Core knowledge retrieval",
    version="1.0.0"
)

# Enable CORS for C# client
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG instance (singleton)
_rag_instance: Optional[RAG] = None


def get_rag_instance() -> RAG:
    """Get or create RAG instance (singleton)"""
    global _rag_instance
    if _rag_instance is None:
        if not SA_RAG_AVAILABLE:
            raise RuntimeError("SA-RAG is not available. Please ensure the Python package is installed.")
        
        # Get configuration from environment variables or use defaults
        llm_provider = os.getenv("SA_RAG_LLM_PROVIDER", "mock")
        embedding_provider = os.getenv("SA_RAG_EMBEDDING_PROVIDER", "mock")
        
        _rag_instance = RAG(
            llm_provider=llm_provider,
            embedding_provider=embedding_provider
        )
        
        print(f"✅ SA-RAG Core initialized: LLM={llm_provider}, Embedding={embedding_provider}")
    return _rag_instance


# Request/Response Models
class RAGQueryRequest(BaseModel):
    """RAG query request model"""
    query: str = Field(..., description="User query text")
    use_graph: bool = Field(True, description="Enable graph expansion")
    use_memory: bool = Field(True, description="Enable memory retrieval")
    top_k: int = Field(6, ge=1, le=50, description="Number of results to return")


class NodeResult(BaseModel):
    """Semantic node result"""
    id: int
    text: str
    score: float
    source: str = "unknown"
    node_type: str = "text"


class ExecutionGraphNode(BaseModel):
    """Execution graph node"""
    node_id: str
    node_type: str
    description: str
    execution_time_ms: float = 0.0


class ExecutionGraphEdge(BaseModel):
    """Execution graph edge"""
    from_node: str
    to_node: str
    edge_type: str
    weight: float = 1.0


class ExecutionGraph(BaseModel):
    """Execution graph structure"""
    query: str
    nodes: List[ExecutionGraphNode] = []
    edges: List[ExecutionGraphEdge] = []
    execution_trace: List[str] = []
    total_time_ms: float = 0.0


class RAGQueryResponse(BaseModel):
    """RAG query response model"""
    answer: str
    nodes: List[NodeResult]
    execution_graph: Optional[ExecutionGraph] = None
    query: str
    top_k: int


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "service": "SA-AgentOS RAG API",
        "version": "1.0.0",
        "status": "running",
        "sa_rag_available": SA_RAG_AVAILABLE
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    try:
        rag = get_rag_instance()
        return {
            "status": "healthy",
            "sa_rag_available": SA_RAG_AVAILABLE,
            "rag_initialized": rag is not None
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e)
        }


@app.post("/rag/query", response_model=RAGQueryResponse)
async def rag_query(request: RAGQueryRequest):
    """
    Execute RAG query
    
    Args:
        request: RAG query request with query text and options
        
    Returns:
        RAG query response with answer, nodes, and execution graph
    """
    if not SA_RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SA-RAG is not available. Please ensure the Python package is installed."
        )
    
    try:
        rag = get_rag_instance()
        
        # Execute search
        search_results = rag.search(
            query=request.query,
            top_k=request.top_k,
            use_graph=request.use_graph,
            use_memory=request.use_memory
        )
        
        # Convert search results to NodeResult format
        nodes: List[NodeResult] = []
        for i, result in enumerate(search_results):
            if isinstance(result, dict):
                node = NodeResult(
                    id=result.get("node_id", i),
                    text=result.get("text", result.get("content", "")),
                    score=result.get("score", 0.0),
                    source=result.get("source", "search"),
                    node_type=result.get("node_type", "text")
                )
            else:
                # Fallback for different result formats
                node = NodeResult(
                    id=i,
                    text=str(result),
                    score=0.5,
                    source="search",
                    node_type="text"
                )
            nodes.append(node)
        
        # Execute Q&A to get answer
        answer_result = rag.ask(
            query=request.query,
            top_k=request.top_k,
            use_graph=request.use_graph,
            use_memory=request.use_memory
        )
        
        answer = ""
        if isinstance(answer_result, dict):
            answer = answer_result.get("answer", answer_result.get("text", answer_result.get("response", "")))
        else:
            answer = str(answer_result)
        
        # Try to get execution graph (if available)
        execution_graph: Optional[ExecutionGraph] = None
        try:
            from sa_rag.execution_graph import ExecutionGraphBuilder
            builder = ExecutionGraphBuilder()
            graph_data = builder.build_graph(
                query=request.query,
                intent="query",
                knowledge_types=["general"]
            )
            
            execution_graph = ExecutionGraph(
                query=request.query,
                nodes=[
                    ExecutionGraphNode(
                        node_id=node.get("node_id", ""),
                        node_type=node.get("node_type", ""),
                        description=node.get("description", ""),
                        execution_time_ms=node.get("execution_time_ms", 0.0)
                    )
                    for node in graph_data.get("nodes", [])
                ],
                edges=[
                    ExecutionGraphEdge(
                        from_node=edge.get("from_node", ""),
                        to_node=edge.get("to_node", ""),
                        edge_type=edge.get("edge_type", "data_flow"),
                        weight=edge.get("weight", 1.0)
                    )
                    for edge in graph_data.get("edges", [])
                ],
                execution_trace=graph_data.get("execution_trace", []),
                total_time_ms=graph_data.get("total_time_ms", 0.0)
            )
        except Exception as e:
            # Execution graph is optional, log but don't fail
            print(f"Warning: Could not build execution graph: {e}")
        
        return RAGQueryResponse(
            answer=answer,
            nodes=nodes,
            execution_graph=execution_graph,
            query=request.query,
            top_k=request.top_k
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query failed: {str(e)}"
        )


@app.post("/rag/memory")
async def add_memory(text: str, importance: float = 0.5):
    """
    Add memory to SA-RAG
    
    Args:
        text: Memory content
        importance: Importance score (0.0-1.0)
        
    Returns:
        Success status
    """
    if not SA_RAG_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="SA-RAG is not available"
        )
    
    try:
        rag = get_rag_instance()
        rag.add_memory(text, importance)
        return {"status": "success", "message": "Memory added"}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to add memory: {str(e)}"
        )


if __name__ == "__main__":
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=True
    )

