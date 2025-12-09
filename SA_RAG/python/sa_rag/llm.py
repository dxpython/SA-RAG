"""LLM Service Module

Provides multiple large language model interfaces:
- OpenAI GPT series
- DeepSeek
- Local models (Llama, etc.)
- Mock mode (for testing)
"""

import os
from typing import List, Optional, Dict, Any
import json


class LLMService:
    """Large Language Model Service"""
    
    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.7,
    ):
        """
        Initialize LLM service
        
        Args:
            provider: Service provider ("openai", "deepseek", "local", "mock")
            model_name: Model name
            api_key: API key
            base_url: API base URL (for custom endpoints)
            temperature: Generation temperature
        """
        self.provider = provider.lower()
        self.model_name = model_name or self._get_default_model()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        self.base_url = base_url
        self.temperature = temperature
        self.mock_mode = provider == "mock" or not self.api_key
        
    def _get_default_model(self) -> str:
        """Get default model name"""
        if self.provider == "openai":
            return "gpt-3.5-turbo"
        elif self.provider == "deepseek":
            return "deepseek-chat"
        else:
            return "mock"
    
    def chat_completion(
        self,
        prompt: str,
        system_prompt: str = "You are a helpful assistant.",
        context: Optional[List[Dict[str, str]]] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """
        Generate chat completion
        
        Args:
            prompt: User prompt
            system_prompt: System prompt
            context: Context message list (optional)
            max_tokens: Maximum tokens to generate
            
        Returns:
            Generated text
        """
        if self.mock_mode:
            return self._mock_chat_completion(prompt, system_prompt, context)
        
        if self.provider == "openai":
            return self._openai_chat_completion(prompt, system_prompt, context, max_tokens)
        elif self.provider == "deepseek":
            return self._deepseek_chat_completion(prompt, system_prompt, context, max_tokens)
        elif self.provider == "local":
            return self._local_chat_completion(prompt, system_prompt, context, max_tokens)
        else:
            return self._mock_chat_completion(prompt, system_prompt, context)
    
    def _openai_chat_completion(
        self,
        prompt: str,
        system_prompt: str,
        context: Optional[List[Dict[str, str]]],
        max_tokens: Optional[int],
    ) -> str:
        """Generate completion using OpenAI API"""
        try:
            import openai
            client = openai.OpenAI(
                api_key=self.api_key,
                base_url=self.base_url,
            )
            
            messages = [{"role": "system", "content": system_prompt}]
            
            if context:
                messages.extend(context)
            
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                max_tokens=max_tokens,
            )
            
            return response.choices[0].message.content or ""
        except ImportError:
            print("Warning: openai package not installed, using mock response")
            return self._mock_chat_completion(prompt, system_prompt, context)
        except Exception as e:
            print(f"Error calling OpenAI API: {e}, using mock response")
            return self._mock_chat_completion(prompt, system_prompt, context)
    
    def _deepseek_chat_completion(
        self,
        prompt: str,
        system_prompt: str,
        context: Optional[List[Dict[str, str]]],
        max_tokens: Optional[int],
    ) -> str:
        """Generate completion using DeepSeek API"""
        try:
            import requests
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }
            
            messages = [{"role": "system", "content": system_prompt}]
            if context:
                messages.extend(context)
            messages.append({"role": "user", "content": prompt})
            
            data = {
                "model": self.model_name,
                "messages": messages,
                "temperature": self.temperature,
            }
            if max_tokens:
                data["max_tokens"] = max_tokens
            
            response = requests.post(
                "https://api.deepseek.com/v1/chat/completions",
                headers=headers,
                json=data
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except ImportError:
            print("Warning: requests package not installed, using mock response")
            return self._mock_chat_completion(prompt, system_prompt, context)
        except Exception as e:
            print(f"Error calling DeepSeek API: {e}, using mock response")
            return self._mock_chat_completion(prompt, system_prompt, context)
    
    def _local_chat_completion(
        self,
        prompt: str,
        system_prompt: str,
        context: Optional[List[Dict[str, str]]],
        max_tokens: Optional[int],
    ) -> str:
        """Generate completion using local model"""
        # Can integrate llama.cpp, transformers, etc. here
        # Example implementation (needs adjustment based on actual library used)
        try:
            # Example: using transformers
            # from transformers import pipeline
            # generator = pipeline("text-generation", model=self.model_name)
            # result = generator(prompt, max_length=max_tokens or 512)
            # return result[0]["generated_text"]
            print("Local model not configured, using mock response")
            return self._mock_chat_completion(prompt, system_prompt, context)
        except Exception as e:
            print(f"Error with local model: {e}, using mock response")
            return self._mock_chat_completion(prompt, system_prompt, context)
    
    def _mock_chat_completion(
        self,
        prompt: str,
        system_prompt: str,
        context: Optional[List[Dict[str, str]]],
    ) -> str:
        """Mock LLM response (for testing)"""
        # Simple mock: generate response based on prompt
        context_str = ""
        if context:
            context_str = "\n".join([msg.get("content", "") for msg in context])
        
        # Mock response: extract key information and generate simple answer
        response = f"Based on the provided content, I understand your question is about: {prompt[:50]}...\n\n"
        if context_str:
            response += f"Based on the context information, I can provide the following answer:\n"
            response += f"Relevant content summary: {context_str[:200]}...\n\n"
        response += "This is a mock response. In actual use, please configure a real LLM API."
        
        return response
    
    def generate_with_rag(
        self,
        query: str,
        retrieved_context: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
    ) -> str:
        """
        Generate answer based on RAG retrieval results
        
        Args:
            query: User query
            retrieved_context: Retrieved context (contains text, score, etc.)
            system_prompt: System prompt (optional)
            
        Returns:
            Generated answer
        """
        system = system_prompt or (
            "You are a helpful assistant that answers questions based on the provided context. "
            "Use only the information from the context to answer. If the context doesn't contain "
            "enough information, say so."
        )
        
        # Build context
        context_text = "\n\n".join([
            f"[Document {i+1}, Score: {ctx.get('score', 0):.3f}]\n{ctx.get('text', '')}"
            for i, ctx in enumerate(retrieved_context)
        ])
        
        prompt = f"""Context:
{context_text}

Question: {query}

Please provide a comprehensive answer based on the context above."""
        
        return self.chat_completion(prompt, system)
