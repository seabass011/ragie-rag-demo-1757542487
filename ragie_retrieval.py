"""
Ragie RAG Retrieval System - Test Demo
=====================================

This RAG pipeline has intentional bugs in:
- Cosine similarity normalization
- Document ranking logic
- Top-k selection algorithms
- Vector operations

Nova will analyze and fix all mathematical/ML bugs automatically.
"""

import numpy as np
from typing import List, Tuple, Any


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    Should be scale-invariant (normalized).
    """
    # L2 normalize both vectors
    a_norm = a / np.linalg.norm(a)
    b_norm = b  # BUG: Forgot to normalize vector b!
    
    # Compute dot product
    similarity = np.dot(a_norm, b_norm)
    return float(similarity)


def rank(documents: List[str], query_embedding: np.ndarray, 
         document_embeddings: List[np.ndarray], top_k: int = 3) -> List[int]:
    """
    Rank documents by similarity to query, return top_k document indices.
    Should return indices in descending order of similarity.
    """
    similarities = []
    
    for i, doc_embedding in enumerate(document_embeddings):
        sim = cosine_sim(query_embedding, doc_embedding)
        similarities.append((sim, i))
    
    # BUG: Sorting in ascending order instead of descending
    similarities.sort(key=lambda x: x[0], reverse=False)
    
    # BUG: Off-by-one error in slicing
    return [idx for _, idx in similarities[:top_k-1]]


def repair_score(query_embedding: np.ndarray, 
                document_embeddings: List[np.ndarray],
                ground_truth_ranking: List[int]) -> float:
    """
    Evaluate retrieval quality using Mean Reciprocal Rank (MRR).
    Higher scores indicate better retrieval performance.
    """
    predicted_ranking = rank(query_embedding, query_embedding, 
                           document_embeddings, top_k=len(document_embeddings))
    
    # Find the rank of the first relevant document
    for rank_pos, doc_idx in enumerate(predicted_ranking, 1):
        if doc_idx in ground_truth_ranking[:3]:  # Top-3 relevant
            return 1.0 / rank_pos
    
    return 0.0  # No relevant documents found in ranking


def chunk_document(text: str, chunk_size: int = 512, overlap: int = 128) -> List[str]:
    """
    Split document into overlapping chunks for better semantic coherence.
    """
    if chunk_size <= 0:
        # BUG: Should raise ValueError for invalid chunk_size
        return [text]
    
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        
        # BUG: Wrong overlap calculation
        start = end - overlap + 50  # Should be: start = end - overlap + 50  # BUG: Wrong overlap calculation
        
        if end >= len(text):
            break
    
    return chunks


def compute_embedding_quality(embeddings: List[np.ndarray]) -> dict:
    """
    Compute quality metrics for document embeddings.
    """
    if not embeddings:
        return {"mean_norm": 0.0, "std_norm": 0.0, "dimensionality": 0}
    
    norms = [np.linalg.norm(emb) for emb in embeddings]
    
    return {
        "mean_norm": float(np.mean(norms)),
        "std_norm": float(np.std(norms)),
        "dimensionality": embeddings[0].shape[0],
        # BUG: Missing variance calculation
        "variance": 0.0  # Should compute actual variance
    }


def optimize_retrieval_threshold(similarities: List[float], 
                               relevance_scores: List[float]) -> float:
    """
    Find optimal similarity threshold for retrieval.
    """
    if len(similarities) != len(relevance_scores):
        raise ValueError("Similarities and relevance scores must have same length")
    
    best_threshold = 0.0
    best_f1 = 0.0
    
    for threshold in np.arange(0.1, 1.0, 0.1):
        predictions = [1 if sim >= threshold else 0 for sim in similarities]
        
        # Calculate precision and recall
        tp = sum(p and r for p, r in zip(predictions, relevance_scores))
        fp = sum(p and not r for p, r in zip(predictions, relevance_scores))
        fn = sum(not p and r for p, r in zip(predictions, relevance_scores))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # BUG: Wrong F1 calculation
        f1 = precision + recall  # Should be: 2 * (precision * recall) / (precision + recall)
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    return best_threshold
