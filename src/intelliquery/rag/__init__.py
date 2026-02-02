"""RAG Module - Document processing and search"""

from .document_processor import (
    process_document,
    answer_question,
    get_document_stats,
    chunk_text,
    cosine_similarity
)

__all__ = [
    "process_document",
    "answer_question",
    "get_document_stats",
    "chunk_text",
    "cosine_similarity"
]
