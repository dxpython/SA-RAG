"""Query Analysis Module: Query type inference and structured query generation

Provides:
- Query intent classification
- Entity extraction
- Structured query format
- Query type-based retrieval strategy selection
"""

from typing import List, Dict, Any, Optional
from enum import Enum
from dataclasses import dataclass
from .llm import LLMService


class QueryIntent(Enum):
    """Query intent types"""
    DEFINITION = "definition"
    COMPARISON = "comparison"
    PROCEDURE = "procedure"
    FACTUAL = "factual"
    CONCEPTUAL = "conceptual"
    OTHER = "other"


@dataclass
class StructuredQuery:
    """Structured query representation"""
    original_query: str
    rewritten_query: str
    intent: QueryIntent
    entities: List[str]
    keywords: List[str]
    relationships: List[str]  # Relationships to search for
    requires_graph: bool
    requires_memory: bool
    top_k: int


class QueryAnalyzer:
    """Query analyzer for intent classification and entity extraction"""
    
    def __init__(self, llm_service: Optional[LLMService] = None):
        """
        Initialize query analyzer
        
        Args:
            llm_service: LLM service for advanced analysis (optional)
        """
        self.llm = llm_service
    
    def analyze(self, query: str, use_llm: bool = False) -> StructuredQuery:
        """
        Analyze query and generate structured query
        
        Args:
            query: User query
            use_llm: Whether to use LLM for analysis
            
        Returns:
            StructuredQuery object
        """
        # Extract keywords
        keywords = self._extract_keywords(query)
        
        # Classify intent
        intent = self._classify_intent(query, use_llm)
        
        # Extract entities
        entities = self._extract_entities(query, use_llm)
        
        # Detect relationships
        relationships = self._detect_relationships(query)
        
        # Determine retrieval requirements
        requires_graph = self._requires_graph(query, intent, relationships)
        requires_memory = self._requires_memory(query)
        
        # Rewrite query
        rewritten_query = self._rewrite_query(query, intent, use_llm)
        
        # Determine top_k based on intent
        top_k = self._determine_top_k(intent)
        
        return StructuredQuery(
            original_query=query,
            rewritten_query=rewritten_query,
            intent=intent,
            entities=entities,
            keywords=keywords,
            relationships=relationships,
            requires_graph=requires_graph,
            requires_memory=requires_memory,
            top_k=top_k,
        )
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query"""
        # Simple keyword extraction
        stop_words = {
            "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
            "of", "with", "by", "is", "are", "was", "were", "be", "been",
            "have", "has", "had", "do", "does", "did", "what", "how", "why",
            "when", "where", "who", "which", "this", "that", "these", "those",
        }
        
        keywords = []
        for word in query.lower().split():
            word = word.strip(".,!?;:")
            if len(word) > 2 and word not in stop_words:
                keywords.append(word)
        
        return keywords
    
    def _classify_intent(self, query: str, use_llm: bool) -> QueryIntent:
        """Classify query intent"""
        if use_llm and self.llm:
            return self._classify_intent_llm(query)
        else:
            return self._classify_intent_rule(query)
    
    def _classify_intent_rule(self, query: str) -> QueryIntent:
        """Rule-based intent classification"""
        query_lower = query.lower()
        
        # Definition patterns
        if any(phrase in query_lower for phrase in [
            "what is", "what are", "define", "definition", "means", "refers to"
        ]):
            return QueryIntent.DEFINITION
        
        # Comparison patterns
        if any(phrase in query_lower for phrase in [
            "compare", "difference", "vs", "versus", "better", "worse",
            "similar", "different", "relationship between"
        ]):
            return QueryIntent.COMPARISON
        
        # Procedure patterns
        if any(phrase in query_lower for phrase in [
            "how to", "how do", "steps", "procedure", "process", "algorithm",
            "method", "way to", "tutorial"
        ]):
            return QueryIntent.PROCEDURE
        
        # Factual patterns
        if any(phrase in query_lower for phrase in [
            "when", "where", "who", "which", "how many", "how much"
        ]):
            return QueryIntent.FACTUAL
        
        # Conceptual patterns
        if any(phrase in query_lower for phrase in [
            "explain", "understand", "concept", "theory", "principle"
        ]):
            return QueryIntent.CONCEPTUAL
        
        return QueryIntent.OTHER
    
    def _classify_intent_llm(self, query: str) -> QueryIntent:
        """LLM-based intent classification"""
        prompt = f"""Classify the following query into one of these intents:
- definition: Asking for definition or meaning
- comparison: Comparing two or more things
- procedure: Asking for steps or how-to
- factual: Asking for facts (when, where, who, etc.)
- conceptual: Asking for explanation or understanding
- other: Other types

Query: {query}

Intent:"""
        
        try:
            response = self.llm.chat_completion(
                prompt,
                system_prompt="You are a query classification expert.",
                max_tokens=20,
            )
            response = response.strip().lower()
            
            # Map response to enum
            intent_map = {
                "definition": QueryIntent.DEFINITION,
                "comparison": QueryIntent.COMPARISON,
                "procedure": QueryIntent.PROCEDURE,
                "factual": QueryIntent.FACTUAL,
                "conceptual": QueryIntent.CONCEPTUAL,
            }
            
            return intent_map.get(response, QueryIntent.OTHER)
        except Exception:
            return self._classify_intent_rule(query)
    
    def _extract_entities(self, query: str, use_llm: bool) -> List[str]:
        """Extract entities from query"""
        if use_llm and self.llm:
            return self._extract_entities_llm(query)
        else:
            return self._extract_entities_rule(query)
    
    def _extract_entities_rule(self, query: str) -> List[str]:
        """Rule-based entity extraction (simple: capitalized words)"""
        entities = []
        words = query.split()
        for word in words:
            word_clean = word.strip(".,!?;:")
            if word_clean and word_clean[0].isupper() and len(word_clean) > 1:
                entities.append(word_clean)
        return entities
    
    def _extract_entities_llm(self, query: str) -> List[str]:
        """LLM-based entity extraction"""
        prompt = f"""Extract all entities (names, concepts, objects) from the following query.
Return them as a comma-separated list.

Query: {query}

Entities:"""
        
        try:
            response = self.llm.chat_completion(
                prompt,
                system_prompt="You are an entity extraction expert.",
                max_tokens=100,
            )
            entities = [e.strip() for e in response.split(",")]
            return [e for e in entities if e]
        except Exception:
            return self._extract_entities_rule(query)
    
    def _detect_relationships(self, query: str) -> List[str]:
        """Detect relationships mentioned in query"""
        relationships = []
        query_lower = query.lower()
        
        relationship_keywords = [
            "relationship", "related", "connects", "links", "associates",
            "depends on", "causes", "leads to", "results in"
        ]
        
        for keyword in relationship_keywords:
            if keyword in query_lower:
                relationships.append(keyword)
        
        return relationships
    
    def _requires_graph(self, query: str, intent: QueryIntent, relationships: List[str]) -> bool:
        """Determine if graph expansion is needed"""
        if relationships:
            return True
        
        if intent in [QueryIntent.COMPARISON, QueryIntent.CONCEPTUAL]:
            return True
        
        query_lower = query.lower()
        graph_keywords = [
            "relationship", "related", "connection", "link", "associate",
            "hierarchy", "structure", "graph", "network"
        ]
        
        return any(keyword in query_lower for keyword in graph_keywords)
    
    def _requires_memory(self, query: str) -> bool:
        """Determine if memory retrieval is needed"""
        query_lower = query.lower()
        memory_keywords = [
            "before", "previous", "earlier", "remember", "mentioned",
            "discussed", "history", "past", "earlier"
        ]
        
        return any(keyword in query_lower for keyword in memory_keywords)
    
    def _rewrite_query(self, query: str, intent: QueryIntent, use_llm: bool) -> str:
        """Rewrite query based on intent"""
        if use_llm and self.llm:
            return self._rewrite_query_llm(query, intent)
        else:
            return self._rewrite_query_rule(query, intent)
    
    def _rewrite_query_rule(self, query: str, intent: QueryIntent) -> str:
        """Rule-based query rewriting"""
        # Simple normalization
        query = " ".join(query.split())
        
        # Intent-specific rewriting
        if intent == QueryIntent.DEFINITION:
            # Ensure definition queries are clear
            if not query.lower().startswith(("what is", "what are", "define")):
                query = f"definition of {query}"
        
        return query
    
    def _rewrite_query_llm(self, query: str, intent: QueryIntent) -> str:
        """LLM-based query rewriting"""
        intent_str = intent.value
        prompt = f"""Rewrite the following query to be optimal for {intent_str} retrieval.
Preserve the core meaning but make it more suitable for information retrieval.

Original query: {query}

Rewritten query:"""
        
        try:
            response = self.llm.chat_completion(
                prompt,
                system_prompt="You are a query rewriting expert for information retrieval.",
                max_tokens=200,
            )
            return response.strip()
        except Exception:
            return self._rewrite_query_rule(query, intent)
    
    def _determine_top_k(self, intent: QueryIntent) -> int:
        """Determine top_k based on intent"""
        intent_top_k = {
            QueryIntent.DEFINITION: 3,
            QueryIntent.COMPARISON: 5,
            QueryIntent.PROCEDURE: 10,
            QueryIntent.FACTUAL: 3,
            QueryIntent.CONCEPTUAL: 5,
            QueryIntent.OTHER: 5,
        }
        return intent_top_k.get(intent, 5)

