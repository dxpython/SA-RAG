"""Embedding Vector Generation Module

Provides multiple embedding service interfaces:
- OpenAI embeddings
- DeepSeek embeddings
- Local model embeddings
- Vector normalization utilities
"""

import os
from typing import List, Optional, Union, Dict, Dict
import hashlib
import math


class EmbeddingService:
    """Embedding vector generation service"""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        dimension: int = 1536,
        enable_cache: bool = True,
        fallback_providers: Optional[List[str]] = None,
    ):
        """
        Initialize embedding service
        
        Args:
            provider: Service provider ("openai", "deepseek", "local", "mock")
            model_name: Model name
            api_key: API key
            dimension: Vector dimension
            enable_cache: Whether to enable embedding cache
            fallback_providers: List of fallback providers in order
        """
        self.provider = provider.lower()
        self.model_name = model_name or self._get_default_model()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.dimension = dimension
        self.mock_mode = provider == "mock" or not self.api_key
        self.enable_cache = enable_cache
        self.cache: Dict[str, List[float]] = {}  # text_hash -> embedding
        self.fallback_providers = fallback_providers or ["deepseek", "openai", "local", "mock"]
        
    def _get_default_model(self) -> str:
        """Get default model name"""
        if self.provider == "openai":
            return "text-embedding-3-small"
        elif self.provider == "deepseek":
            return "deepseek-embedding"
        else:
            return "mock"
    
    def get_embedding(self, text: str) -> List[float]:
        """
        Generate embedding vector for text (with caching and fallback)
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector list
        """
        # Check cache
        if self.enable_cache:
            text_hash = self._hash_text(text)
            if text_hash in self.cache:
                return self.cache[text_hash]
        
        # Get embedding
        if self.mock_mode:
            embedding = self._generate_mock_embedding(text)
        elif self.provider == "openai":
            embedding = self._get_openai_embedding(text)
        elif self.provider == "deepseek":
            embedding = self._get_deepseek_embedding(text)
        elif self.provider == "local":
            embedding = self._get_local_embedding(text)
        else:
            embedding = self._generate_mock_embedding(text)
        
        # Cache result
        if self.enable_cache:
            text_hash = self._hash_text(text)
            self.cache[text_hash] = embedding
        
        return embedding
    
    def get_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings in batch
        
        Args:
            texts: List of texts
            
        Returns:
            List of embedding vectors
        """
        if self.mock_mode:
            return [self._generate_mock_embedding(text) for text in texts]
        
        if self.provider == "openai":
            return self._get_openai_embeddings_batch(texts)
        elif self.provider == "deepseek":
            return self._get_deepseek_embeddings_batch(texts)
        else:
            return [self._generate_mock_embedding(text) for text in texts]
    
    def _get_openai_embedding(self, text: str) -> List[float]:
        """Generate embedding using OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model_name,
                input=text
            )
            return response.data[0].embedding
        except ImportError:
            print("Warning: openai package not installed, using mock embedding")
            return self._generate_mock_embedding(text)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}, using mock embedding")
            return self._generate_mock_embedding(text)
    
    def _get_openai_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch using OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key)
            response = client.embeddings.create(
                model=self.model_name,
                input=texts
            )
            return [item.embedding for item in response.data]
        except ImportError:
            return [self._generate_mock_embedding(text) for text in texts]
        except Exception as e:
            print(f"Error calling OpenAI API: {e}, using mock embeddings")
            return [self._generate_mock_embedding(text) for text in texts]
    
    def _get_deepseek_embedding(self, text: str) -> List[float]:
        """Generate embedding using DeepSeek API"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "input": text
            }
            response = requests.post(
                "https://api.deepseek.com/v1/embeddings",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["data"][0]["embedding"]
        except ImportError:
            print("Warning: requests package not installed, using mock embedding")
            return self._generate_mock_embedding(text)
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}, using mock embedding")
            return self._generate_mock_embedding(text)
    
    def _get_deepseek_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings in batch using DeepSeek API"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            data = {
                "model": self.model_name,
                "input": texts
            }
            response = requests.post(
                "https://api.deepseek.com/v1/embeddings",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return [item["embedding"] for item in response.json()["data"]]
        except ImportError:
            return [self._generate_mock_embedding(text) for text in texts]
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}, using mock embeddings")
            return [self._generate_mock_embedding(text) for text in texts]
    
    def _get_local_embedding(self, text: str) -> List[float]:
        """Generate embedding using local model (requires installing corresponding library)"""
        # Can integrate sentence-transformers or other local models here
        # Example: using sentence-transformers
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            embedding = model.encode(text, convert_to_numpy=False)
            return embedding.tolist() if hasattr(embedding, 'tolist') else list(embedding)
        except ImportError:
            print("Warning: sentence-transformers not installed, using mock embedding")
            return self._generate_mock_embedding(text)
        except Exception as e:
            print(f"Error with local model: {e}, using mock embedding")
            return self._generate_mock_embedding(text)
    
    def _generate_mock_embedding(self, text: str) -> List[float]:
        """
        Generate mock embedding vector (deterministic vector based on text hash)
        
        For testing and demonstration, does not depend on external API
        """
        # Use text hash to generate deterministic vector
        hash_obj = hashlib.sha256(text.encode('utf-8'))
        hash_bytes = hash_obj.digest()
        
        # Convert hash to float vector
        embedding = []
        for i in range(0, min(len(hash_bytes), self.dimension * 4), 4):
            # Combine 4 bytes into a float
            value = int.from_bytes(hash_bytes[i:i+4], byteorder='big')
            # Normalize to [-1, 1] range
            normalized = (value / (2**32 - 1)) * 2 - 1
            embedding.append(normalized)
        
        # If dimension is insufficient, fill with hash values
        while len(embedding) < self.dimension:
            hash_obj.update(str(len(embedding)).encode('utf-8'))
            hash_bytes = hash_obj.digest()
            value = int.from_bytes(hash_bytes[:4], byteorder='big')
            normalized = (value / (2**32 - 1)) * 2 - 1
            embedding.append(normalized)
        
        # Truncate to specified dimension
        embedding = embedding[:self.dimension]
        
        # L2 normalization
        return self.normalize(embedding)
    
    @staticmethod
    def normalize(vec: List[float]) -> List[float]:
        """
        L2 normalize vector
        
        Args:
            vec: Input vector
            
        Returns:
            Normalized vector
        """
        norm = math.sqrt(sum(x * x for x in vec))
        if norm < 1e-10:
            return vec
        return [x / norm for x in vec]
    
    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """
        Calculate cosine similarity between two vectors
        
        Args:
            vec1: Vector 1
            vec2: Vector 2
            
        Returns:
            Cosine similarity (-1 to 1)
        """
        if len(vec1) != len(vec2):
            return 0.0
        
        dot_product = sum(x * y for x, y in zip(vec1, vec2))
        norm1 = math.sqrt(sum(x * x for x in vec1))
        norm2 = math.sqrt(sum(x * x for x in vec2))
        
        if norm1 < 1e-10 or norm2 < 1e-10:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _hash_text(self, text: str) -> str:
        """Generate hash for text (for caching)"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def clear_cache(self):
        """Clear embedding cache"""
        if hasattr(self, 'cache'):
            self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Union[int, str]]:
        """Get cache statistics"""
        return {
            "size": len(self.cache) if hasattr(self, 'cache') else 0,
            "provider": self.provider,
        }
