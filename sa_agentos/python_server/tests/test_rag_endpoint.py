"""
Tests for SA-AgentOS RAG API endpoints
"""

import pytest
from fastapi.testclient import TestClient
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from server import app

client = TestClient(app)


def test_root_endpoint():
    """Test root endpoint"""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["service"] == "SA-AgentOS RAG API"


def test_health_endpoint():
    """Test health check endpoint"""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data


def test_rag_query_success():
    """Test successful RAG query"""
    response = client.post(
        "/rag/query",
        json={
            "query": "What is Python?",
            "use_graph": True,
            "use_memory": True,
            "top_k": 5
        }
    )
    
    # Should return 200 even if SA-RAG is not fully initialized
    # (it will use mock mode)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "answer" in data
        assert "nodes" in data
        assert "query" in data
        assert isinstance(data["nodes"], list)
        assert data["query"] == "What is Python?"


def test_rag_query_missing_query():
    """Test RAG query with missing query field"""
    response = client.post(
        "/rag/query",
        json={
            "use_graph": True,
            "top_k": 5
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_rag_query_invalid_top_k():
    """Test RAG query with invalid top_k"""
    response = client.post(
        "/rag/query",
        json={
            "query": "test",
            "top_k": 0  # Invalid: must be >= 1
        }
    )
    
    assert response.status_code == 422  # Validation error


def test_rag_query_defaults():
    """Test RAG query with default values"""
    response = client.post(
        "/rag/query",
        json={
            "query": "test query"
        }
    )
    
    # Should accept with defaults
    assert response.status_code in [200, 503]


def test_add_memory():
    """Test adding memory"""
    response = client.post(
        "/rag/memory",
        params={
            "text": "User prefers Python",
            "importance": 0.8
        }
    )
    
    # Should return 200 or 503 (if SA-RAG not available)
    assert response.status_code in [200, 503]
    
    if response.status_code == 200:
        data = response.json()
        assert "status" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

