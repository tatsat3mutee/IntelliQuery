"""
Query Router - Intelligent Query Classification and Routing
============================================================
Classifies user queries and routes them to the appropriate handler:
- KNOWLEDGE: Document RAG for conceptual questions
- DATA: Text-to-SQL for data queries
- HYBRID: Combines both for complex questions

Features:
- Keyword-based classification
- Confidence scoring
- Query intent extraction
"""

import logging
import re
from typing import Dict, Any, Tuple
from enum import Enum
from dataclasses import dataclass

from ..rag.document_processor import answer_question as rag_answer_question
from ..analytics.text_to_sql import text_to_sql_agent

logger = logging.getLogger(__name__)


class QueryType(Enum):
    """Types of queries we handle"""
    KNOWLEDGE = "knowledge"   # Document/conceptual questions
    DATA = "data"            # Data/metrics questions
    HYBRID = "hybrid"        # Need both document context and data
    UNKNOWN = "unknown"      # Cannot classify


@dataclass  
class ClassificationResult:
    """Result of query classification"""
    query_type: QueryType
    confidence: float
    keywords_matched: list
    intent: str


class QueryRouter:
    """Router for classifying and handling queries"""
    
    # Keywords that indicate DATA queries
    DATA_KEYWORDS = [
        # Aggregation words
        'how many', 'count', 'number of', 'total number', 'total count',
        'average', 'avg', 'mean',
        'sum', 'sum of', 'total revenue', 'total cost',
        'maximum', 'max', 'highest', 'largest', 'top',
        'minimum', 'min', 'lowest', 'smallest', 'bottom',
        
        # Data-specific nouns
        'customers', 'customer', 'records', 'rows',
        'churned', 'churn rate', 'retention',
        'revenue', 'charges', 'monthly charge',
        'tenure', 'months',
        
        # Query action words
        'list all', 'show all', 'display', 'get all',
        'breakdown', 'by contract', 'by gender', 'per',
        'filter', 'where', 'who has',
        
        # Statistics
        'statistics', 'stats', 'metrics', 'kpi',
    ]
    
    # Keywords that indicate KNOWLEDGE queries
    KNOWLEDGE_KEYWORDS = [
        # Question words for concepts
        'what is', 'what are', "what's",
        'explain', 'explain how', 'explain why',
        'how does', 'how do', 'how can',
        'why is', 'why does', 'why do',
        'describe', 'definition', 'define',
        'tell me about', 'what do you know about',
        'meaning of', 'concept of',
        
        # Document references
        'according to', 'based on the document', 'in the paper',
        'the document says', 'the article', 'the text',
        
        # Conceptual terms
        'mechanism', 'process', 'method', 'approach',
        'algorithm', 'technique', 'strategy',
        'theory', 'principle', 'concept',
        'architecture', 'model', 'framework',
        'attention', 'transformer', 'neural', 'learning',
    ]
    
    # Keywords that indicate HYBRID queries
    HYBRID_KEYWORDS = [
        'based on our data', 'according to both',
        'compare', 'correlation', 'relationship between',
        'why are customers churning', 'explain our churn',
        'analyze', 'insight', 'pattern',
    ]
    
    def __init__(self):
        self.last_classification: ClassificationResult = None
    
    def classify_query(self, question: str) -> ClassificationResult:
        """
        Classify a query into KNOWLEDGE, DATA, or HYBRID
        
        Returns ClassificationResult with:
        - query_type: The classification
        - confidence: 0.0 to 1.0
        - keywords_matched: Which keywords triggered this
        - intent: Brief description of detected intent
        """
        question_lower = question.lower().strip()
        
        # Score each type
        data_score = 0
        knowledge_score = 0
        hybrid_score = 0
        
        data_matches = []
        knowledge_matches = []
        hybrid_matches = []
        
        # Check data keywords
        for keyword in self.DATA_KEYWORDS:
            if keyword in question_lower:
                data_score += 1
                data_matches.append(keyword)
        
        # Check knowledge keywords
        for keyword in self.KNOWLEDGE_KEYWORDS:
            if keyword in question_lower:
                knowledge_score += 1
                knowledge_matches.append(keyword)
        
        # Check hybrid keywords
        for keyword in self.HYBRID_KEYWORDS:
            if keyword in question_lower:
                hybrid_score += 2  # Weight hybrid higher when detected
                hybrid_matches.append(keyword)
        
        # Normalize scores
        total_score = data_score + knowledge_score + hybrid_score
        if total_score == 0:
            # No keywords matched - use heuristics
            # Questions starting with "how many" or containing numbers are likely data
            if re.match(r'^how many', question_lower) or re.search(r'\d+', question_lower):
                return ClassificationResult(
                    query_type=QueryType.DATA,
                    confidence=0.5,
                    keywords_matched=['heuristic: numeric question'],
                    intent="Data query (heuristic)"
                )
            # Questions starting with "what is" are likely knowledge
            if re.match(r'^what (is|are)', question_lower):
                return ClassificationResult(
                    query_type=QueryType.KNOWLEDGE,
                    confidence=0.5,
                    keywords_matched=['heuristic: what is question'],
                    intent="Knowledge query (heuristic)"
                )
            # Default to UNKNOWN
            return ClassificationResult(
                query_type=QueryType.UNKNOWN,
                confidence=0.0,
                keywords_matched=[],
                intent="Could not determine query type"
            )
        
        # Determine winning type
        if hybrid_score > 0 and hybrid_score >= max(data_score, knowledge_score):
            query_type = QueryType.HYBRID
            confidence = hybrid_score / total_score
            matches = hybrid_matches
            intent = "Hybrid query (needs both data and document context)"
        elif data_score > knowledge_score:
            query_type = QueryType.DATA
            confidence = data_score / total_score
            matches = data_matches
            intent = self._infer_data_intent(question_lower, data_matches)
        elif knowledge_score > data_score:
            query_type = QueryType.KNOWLEDGE
            confidence = knowledge_score / total_score
            matches = knowledge_matches
            intent = self._infer_knowledge_intent(question_lower, knowledge_matches)
        else:
            # Tie - default to data if numbers present, else knowledge
            if re.search(r'\d+|customer|churn|revenue', question_lower):
                query_type = QueryType.DATA
                confidence = 0.5
                matches = data_matches
                intent = "Data query (tie-breaker)"
            else:
                query_type = QueryType.KNOWLEDGE
                confidence = 0.5
                matches = knowledge_matches
                intent = "Knowledge query (tie-breaker)"
        
        result = ClassificationResult(
            query_type=query_type,
            confidence=min(confidence, 1.0),
            keywords_matched=matches[:5],  # Limit to top 5
            intent=intent
        )
        
        self.last_classification = result
        logger.info(f"Classified as {query_type.value} (confidence: {confidence:.2f})")
        
        return result
    
    def _infer_data_intent(self, question: str, matches: list) -> str:
        """Infer specific intent for data queries"""
        if 'how many' in question or 'count' in matches:
            return "Count/aggregate data query"
        if 'average' in question or 'avg' in matches:
            return "Average calculation query"
        if 'top' in question or 'highest' in question:
            return "Top/ranking query"
        if 'list' in question or 'show' in question:
            return "List/display query"
        if 'churn' in question:
            return "Churn analysis query"
        return "General data query"
    
    def _infer_knowledge_intent(self, question: str, matches: list) -> str:
        """Infer specific intent for knowledge queries"""
        if 'what is' in question or 'definition' in matches:
            return "Definition/explanation query"
        if 'how does' in question or 'explain' in matches:
            return "Process/mechanism query"
        if 'why' in question:
            return "Reasoning/causation query"
        return "General knowledge query"
    
    def route_query(self, question: str) -> Dict[str, Any]:
        """
        Main entry point: Classify query and route to appropriate handler
        
        Returns:
            {
                "success": bool,
                "query_type": str,
                "confidence": float,
                "intent": str,
                "answer": str,
                "sources": list,        # For knowledge queries
                "sql": str,             # For data queries
                "results": list,        # For data queries
                "error": str            # If failed
            }
        """
        try:
            # Classify the query
            classification = self.classify_query(question)
            
            # Base response
            response = {
                "success": True,
                "query_type": classification.query_type.value,
                "confidence": classification.confidence,
                "intent": classification.intent,
                "keywords_matched": classification.keywords_matched
            }
            
            # Route to appropriate handler
            if classification.query_type == QueryType.DATA:
                # Use Text-to-SQL
                sql_result = text_to_sql_agent.execute_query(question)
                response.update({
                    "answer": sql_result.get("answer", "Could not generate answer"),
                    "sql": sql_result.get("sql"),
                    "results": sql_result.get("results", []),
                    "row_count": sql_result.get("row_count", 0),
                    "columns": sql_result.get("columns", [])
                })
                if not sql_result.get("success"):
                    response["success"] = False
                    response["error"] = sql_result.get("error")
                    
            elif classification.query_type == QueryType.KNOWLEDGE:
                # Use RAG
                rag_result = rag_answer_question(question)
                response.update({
                    "answer": rag_result.get("answer", "Could not find relevant information"),
                    "sources": rag_result.get("sources", [])
                })
                if not rag_result.get("success"):
                    response["success"] = False
                    response["error"] = rag_result.get("error")
                    
            elif classification.query_type == QueryType.HYBRID:
                # Combine both - get RAG context and data
                rag_result = rag_answer_question(question)
                sql_result = text_to_sql_agent.execute_query(question)
                
                # Combine answers
                combined_answer = ""
                if rag_result.get("success") and rag_result.get("answer"):
                    combined_answer += f"**Document Context:**\n{rag_result.get('answer', '')}\n\n"
                if sql_result.get("success") and sql_result.get("answer"):
                    combined_answer += f"**Data Analysis:**\n{sql_result.get('answer', '')}"
                
                if not combined_answer:
                    combined_answer = "Could not find relevant information from documents or data."
                
                response.update({
                    "answer": combined_answer,
                    "sources": rag_result.get("sources", []),
                    "sql": sql_result.get("sql"),
                    "results": sql_result.get("results", [])
                })
                
            else:
                # UNKNOWN - try both and see what works
                logger.info("Unknown query type - trying both handlers")
                
                # Try data first (faster)
                sql_result = text_to_sql_agent.execute_query(question)
                if sql_result.get("success") and sql_result.get("row_count", 0) > 0:
                    response.update({
                        "query_type": "data",
                        "answer": sql_result.get("answer"),
                        "sql": sql_result.get("sql"),
                        "results": sql_result.get("results", [])
                    })
                else:
                    # Fall back to RAG
                    rag_result = rag_answer_question(question)
                    response.update({
                        "query_type": "knowledge",
                        "answer": rag_result.get("answer", "I couldn't find relevant information. Please try rephrasing your question."),
                        "sources": rag_result.get("sources", [])
                    })
            
            return response
            
        except Exception as e:
            logger.error(f"Query routing error: {e}")
            return {
                "success": False,
                "error": str(e),
                "query_type": "error",
                "answer": f"An error occurred while processing your question: {str(e)}"
            }
    
    def get_example_queries(self) -> Dict[str, list]:
        """Get example queries for each type to help users"""
        return {
            "data_queries": [
                "How many customers are in the database?",
                "What is the average monthly charge?",
                "How many customers churned?",
                "Top 10 customers by total charges",
                "Show churn rate by contract type",
                "List customers with fiber internet",
            ],
            "knowledge_queries": [
                "What is the attention mechanism?",
                "Explain how transformers work",
                "What are the key concepts in the document?",
                "Describe the methodology used",
            ],
            "hybrid_queries": [
                "Why are customers churning based on the data patterns?",
                "Analyze the relationship between contract type and churn",
            ]
        }


# Global router instance
query_router = QueryRouter()
