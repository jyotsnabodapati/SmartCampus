import os
import sys

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List, Dict, Any
from llama_index.core import VectorStoreIndex

from llama_index.core.schema import NodeWithScore

def retrieve_top_k_chunks(
    index: VectorStoreIndex,
    query: str,
    top_k: int = 3
) -> List[Dict[str, Any]]:
    """
    Retrieves the top-k most relevant text chunks from the vector index for a given query.
    Calculates similarity scores and extracts document source metadata.

    Args:
        index (VectorStoreIndex): Indexed Vector Database.
        query (str): User question string.
        top_k (int): Number of top matching chunks to retrieve (default: 3).

    Returns:
        List[Dict[str, Any]]: List of dictionary results containing:
            - text (str): Content of retrieved chunk
            - score (float): Similarity score (0.0 to 1.0)
            - file_name (str): Source filename
    """
    retriever = index.as_retriever(similarity_top_k=top_k)
    retrieved_nodes: List[NodeWithScore] = retriever.retrieve(query)

    results = []
    for node_with_score in retrieved_nodes:
        node = node_with_score.node
        score = node_with_score.score if node_with_score.score is not None else 0.0
        file_name = node.metadata.get("file_name", "Unknown File")

        results.append({
            "text": node.get_content(),
            "score": round(score, 4),
            "file_name": file_name,
            "node_id": node.node_id
        })

    return results

def test_retrieval(index: VectorStoreIndex, sample_query: str = "What is the minimum attendance required?"):
    """Standalone retrieval tester printing top matching chunks and similarity scores."""
    print(f"\n🔎 Testing Retrieval for Query: '{sample_query}'")
    results = retrieve_top_k_chunks(index, sample_query, top_k=3)
    for idx, r in enumerate(results, start=1):
        print(f"\n--- Result #{idx} [Score: {r['score']}] [Source: {r['file_name']}] ---")
        print(f"{r['text'][:250]}...")
    return results

if __name__ == "__main__":
    from src.ingestion import load_documents
    from src.chunking import split_documents_into_chunks
    from src.embeddings import get_embedding_model
    from src.vector_store import create_vector_index

    docs = load_documents("./data")
    if docs:
        chunks = split_documents_into_chunks(docs)
        embed_mdl = get_embedding_model()
        index = create_vector_index(chunks, embed_mdl)
        test_retrieval(index, "What is the minimum attendance required?")
