"""
Document Handler - Upload, chunk, embed, and search documents
=============================================================
Full RAG implementation with:
- Text chunking (word-based with overlap)
- Vector embeddings (Databricks endpoint or mock)
- Vector similarity search (Databricks Vector Search or in-memory fallback)
- LLM-powered answer generation
"""

import uuid
import logging
import math
from datetime import datetime
from typing import List, Dict, Optional, Tuple

from ..core.database import db_client
from ..core.config import config
from .vector_search import vector_search_manager

logger = logging.getLogger(__name__)


# ============== VECTOR OPERATIONS ==============

def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2 or len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def normalize_vector(vec: List[float]) -> List[float]:
    """Normalize a vector to unit length"""
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0:
        return vec
    return [x / norm for x in vec]


def chunk_text(text: str, chunk_size: Optional[int] = None, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks for better context
    
    Args:
        text: The text to chunk
        chunk_size: Max characters per chunk (default from config)
        overlap: Number of words to overlap between chunks
    """
    if chunk_size is None:
        chunk_size = config.CHUNK_SIZE
    
    # Clean text
    text = text.replace('\r\n', '\n').replace('\r', '\n')
    text = ' '.join(text.split())  # Normalize whitespace
    
    words = text.split()
    chunks = []
    i = 0
    
    while i < len(words):
        current_chunk = []
        current_size = 0
        
        # Build chunk up to chunk_size
        while i < len(words) and current_size < chunk_size:
            word = words[i]
            current_chunk.append(word)
            current_size += len(word) + 1
            i += 1
        
        if current_chunk:
            chunks.append(" ".join(current_chunk))
            
            # Move back by overlap words for next chunk
            if overlap > 0 and i < len(words):
                i = max(0, i - overlap)
    
    logger.info(f"Chunked {len(words)} words into {len(chunks)} chunks")
    return chunks


def chunk_text_semantic(text: str, max_chunk_size: Optional[int] = None) -> List[str]:
    """
    Chunk text by sentences for better semantic coherence
    Falls back to word-based if sentences are too long
    """
    if max_chunk_size is None:
        max_chunk_size = config.CHUNK_SIZE
    
    # Split by sentence boundaries
    import re
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    chunks = []
    current_chunk = []
    current_size = 0
    
    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue
            
        sentence_size = len(sentence)
        
        # If single sentence is too long, chunk by words
        if sentence_size > max_chunk_size:
            if current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_size = 0
            
            # Word-based chunking for this sentence
            words = sentence.split()
            word_chunk = []
            word_size = 0
            for word in words:
                word_chunk.append(word)
                word_size += len(word) + 1
                if word_size >= max_chunk_size:
                    chunks.append(" ".join(word_chunk))
                    word_chunk = []
                    word_size = 0
            if word_chunk:
                current_chunk = word_chunk
                current_size = word_size
        elif current_size + sentence_size + 1 > max_chunk_size:
            # Start new chunk
            if current_chunk:
                chunks.append(" ".join(current_chunk))
            current_chunk = [sentence]
            current_size = sentence_size
        else:
            # Add to current chunk
            current_chunk.append(sentence)
            current_size += sentence_size + 1
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks


def process_document(filename: str, content: str, batch_size: int = 5) -> Dict:
    """
    Process and save document to Databricks
    1. Chunk the text
    2. Generate embeddings for each chunk
    3. Save to rag_documents table (batched for performance)
    
    Args:
        filename: Name of the document
        content: Text content
        batch_size: Number of chunks to insert per SQL statement (default: 5)
    """
    try:
        logger.info(f"=== Starting document upload: {filename} ===")
        logger.info(f"Content length: {len(content)} characters")
        logger.info(f"Target table: {config.RAG_TABLE}")
        
        # Chunk text
        chunks = chunk_text(content, config.CHUNK_SIZE)
        logger.info(f"Processing {filename}: {len(chunks)} chunks")
        
        # Generate document ID for this file
        doc_id = str(uuid.uuid4())
        safe_filename = filename.replace("'", "''")
        
        # Process chunks and prepare batch inserts
        saved_count = 0
        batch_values = []
        
        for i, chunk in enumerate(chunks):
            # Get embedding
            embedding = db_client.get_embedding(chunk)
            
            # Format embedding for SQL
            embedding_str = ",".join(map(str, embedding))
            
            # Escape single quotes in text
            safe_text = chunk.replace("'", "''").replace("\\", "\\\\")
            
            # Generate unique ID for this chunk
            chunk_id = str(uuid.uuid4())
            
            # Add to batch - match correct schema: id, filename, text, embedding, chunk_index, upload_date, metadata
            value_row = f"""(
                '{chunk_id}',
                '{safe_filename}',
                '{safe_text}',
                array({embedding_str}),
                {i},
                current_timestamp(),
                map('source', 'upload', 'doc_id', '{doc_id}')
            )"""
            batch_values.append(value_row)
            
            # Execute batch insert when batch is full or last chunk
            if len(batch_values) >= batch_size or i == len(chunks) - 1:
                sql = f"""
                    INSERT INTO {config.RAG_TABLE} 
                    (id, filename, text, embedding, chunk_index, upload_date, metadata)
                    VALUES {','.join(batch_values)}
                """
                logger.info(f"Executing SQL: {sql[:200]}...")
                db_client.execute(sql)
                saved_count += len(batch_values)
                logger.info(f"Saved batch: {saved_count}/{len(chunks)} chunks")
                batch_values = []
        
        return {
            "success": True,
            "filename": filename,
            "chunks_saved": saved_count,
            "message": f"Document uploaded! Saved {saved_count} chunks in {(saved_count + batch_size - 1) // batch_size} batches."
        }
    
    except Exception as e:
        logger.error(f"Error processing document: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def search_documents(question: str, top_k: int = 5) -> List[Dict]:
    """
    Search for similar documents using VECTOR SEARCH
    
    Automatically uses:
    1. Databricks Vector Search (if configured and available) - FAST, scales to millions
    2. In-memory Python search (fallback) - Works but slow for large datasets
    
    Process:
    1. Generate embedding for the question
    2. Search using Vector Index OR fetch all and compare in Python
    3. Return top-k most similar documents
    
    For large datasets (>1000 docs), configure Databricks Vector Search
    """
    try:
        # Step 1: Get embedding for the question
        question_embedding = db_client.get_embedding(question)
        
        # Step 2: Try Vector Search first (if available)
        if vector_search_manager.is_available() and vector_search_manager.index_exists():
            logger.info("🚀 Using Databricks Vector Search (HIGH PERFORMANCE)")
            try:
                documents = vector_search_manager.search(
                    query_vector=question_embedding,
                    top_k=top_k
                )
                if documents:
                    logger.info(f"✓ Vector Search returned {len(documents)} results")
                    return documents
                else:
                    logger.warning("Vector Search returned no results, falling back to in-memory")
            except Exception as e:
                logger.warning(f"Vector Search failed: {e}, falling back to in-memory search")
        
        # Step 3: Fallback to in-memory search
        logger.info("⚠️ Using in-memory vector search (SLOWER - consider enabling Vector Search)")
        return _in_memory_vector_search(question_embedding, top_k)
    
    except Exception as e:
        logger.error(f"Search error: {e}")
        # Last resort: keyword search
        return _keyword_search_fallback(question, top_k)


def _in_memory_vector_search(question_embedding: List[float], top_k: int = 5) -> List[Dict]:
    """
    In-memory vector search (ORIGINAL METHOD)
    Fetches all documents and compares in Python
    
    ⚠️ Warning: Slow for >1000 documents
    """
    try:
        # Fetch documents with embeddings
        sql = f"""
            SELECT id, filename, text, embedding, chunk_index, upload_date
            FROM {config.RAG_TABLE}
            WHERE embedding IS NOT NULL
            ORDER BY upload_date DESC
            LIMIT 500
        """
        results = db_client.query(sql)
        
        if not results:
            logger.warning("No documents found in database")
            return []
        
        # Step 3: Calculate similarity for each document
        scored_docs = []
        for doc in results:
            doc_embedding = doc.get('embedding')
            
            # Handle different embedding formats
            if doc_embedding is None:
                continue
                
            # Parse embedding if it's a string
            if isinstance(doc_embedding, str):
                try:
                    doc_embedding = [float(x) for x in doc_embedding.strip('[]').split(',')]
                except:
                    continue
            
            # Convert to list if it's another iterable (numpy array, etc)
            try:
                if hasattr(doc_embedding, '__iter__') and not isinstance(doc_embedding, (str, dict)):
                    doc_embedding = list(doc_embedding)
                    if len(doc_embedding) > 0:
                        similarity = cosine_similarity(question_embedding, doc_embedding)
                        scored_docs.append({
                            'id': doc['id'],
                            'filename': doc['filename'],
                            'text': doc['text'],
                            'chunk_index': doc.get('chunk_index', 0),
                            'upload_date': doc['upload_date'],
                            'similarity': similarity
                        })
            except Exception as e:
                logger.warning(f"Skipping doc due to embedding error: {e}")
                continue
        
        # Step 4: Sort by similarity and return top-k
        scored_docs.sort(key=lambda x: x['similarity'], reverse=True)
        
        logger.info(f"Found {len(scored_docs)} documents, returning top {top_k}")
        return scored_docs[:top_k]
    
    except Exception as e:
        logger.error(f"In-memory vector search error: {e}")
        return []


def _keyword_search_fallback(question: str, top_k: int = 5) -> List[Dict]:
    """Fallback to keyword search if vector search fails"""
    try:
        terms = [t.strip().lower() for t in question.split() if len(t) > 2]
        
        if not terms:
            sql = f"""
                SELECT id, filename, text, upload_date
                FROM {config.RAG_TABLE}
                ORDER BY upload_date DESC
                LIMIT {top_k}
            """
        else:
            like_clauses = " OR ".join([f"LOWER(text) LIKE '%{t}%'" for t in terms])
            score_expr = " + ".join([f"CASE WHEN LOWER(text) LIKE '%{t}%' THEN 1 ELSE 0 END" for t in terms])
            
            sql = f"""
                SELECT id, filename, text, upload_date, ({score_expr}) as relevance
                FROM {config.RAG_TABLE}
                WHERE {like_clauses}
                ORDER BY relevance DESC
                LIMIT {top_k}
            """
        
        results = db_client.query(sql)
        return results or []
    
    except Exception as e:
        logger.error(f"Keyword search fallback error: {e}")
        return []


def answer_question(question: str, top_k: int = 5) -> Dict:
    """
    Answer question using RAG (Retrieval Augmented Generation):
    
    1. Search for relevant documents (vector similarity)
    2. Build context from search results
    3. Generate answer using LLM with context
    
    Returns dict with answer, sources, and similarity scores
    """
    try:
        # Search documents using vector similarity
        docs = search_documents(question, top_k)
        
        if not docs:
            return {
                "success": True,
                "answer": "No relevant documents found. Please upload some documents first.",
                "sources": [],
                "method": "no_docs"
            }
        
        # Build context from retrieved documents
        context_parts = []
        sources = []
        for i, doc in enumerate(docs):
            text = doc.get('text', '')[:1500]  # Limit context per doc
            filename = doc.get('filename', 'Unknown')
            similarity = doc.get('similarity', 0)
            
            context_parts.append(f"[Document {i+1} - {filename} (relevance: {similarity:.2f})]:\n{text}")
            sources.append({
                "filename": filename,
                "similarity": float(round(similarity, 3)) if similarity else None  # Convert to Python float
            })
        
        context = "\n\n---\n\n".join(context_parts)
        
        # Generate answer with LLM
        prompt = f"""Based on the following documents, answer the user's question directly and concisely.

DOCUMENTS:
{context}

QUESTION: {question}

Provide a clear, well-formatted answer based only on the information in the documents above. Use quotes when appropriate. If the information is not in the documents, say so.

ANSWER:"""
        
        answer = db_client.generate_answer(prompt, max_tokens=800)
        
        # Ensure answer is a string
        if not isinstance(answer, str):
            logger.warning(f"Answer is not a string: {type(answer)}, converting...")
            answer = str(answer)
        
        logger.info(f"Generated answer length: {len(answer)} characters")
        
        return {
            "success": True,
            "answer": answer,
            "sources": sources,
            "docs_found": len(docs),
            "method": "rag_vector_search"
        }
    
    except Exception as e:
        logger.error(f"Answer error: {e}")
        return {
            "success": False,
            "error": str(e)
        }


def get_document_stats() -> Dict:
    """Get statistics about uploaded documents"""
    try:
        sql = f"""
            SELECT 
                COUNT(*) as total_chunks,
                COUNT(DISTINCT filename) as total_files,
                MIN(upload_date) as first_upload,
                MAX(upload_date) as last_upload
            FROM {config.RAG_TABLE}
        """
        results = db_client.query(sql)
        
        if results:
            stats = results[0]
            # Convert datetime objects to strings for JSON serialization
            if stats.get('first_upload'):
                stats['first_upload'] = str(stats['first_upload'])
            if stats.get('last_upload'):
                stats['last_upload'] = str(stats['last_upload'])
            return {
                "success": True,
                "stats": stats
            }
        return {"success": True, "stats": {"total_chunks": 0, "total_files": 0}}
    
    except Exception as e:
        logger.error(f"Stats error: {e}")
        return {"success": False, "error": str(e)}
