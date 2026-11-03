import pytest
from src.chunking import split_documents_into_chunks
from src.vector_store import create_vector_index
from src.retriever import retrieve_top_k_chunks


def test_retrieve_top_k_chunks(sample_documents, fast_embed_model, temp_chroma_dir):
    """Verify top-k chunk retrieval returns proper structure and metadata."""
    nodes = split_documents_into_chunks(sample_documents)
    index = create_vector_index(
        nodes=nodes,
        embed_model=fast_embed_model,
        db_path=temp_chroma_dir,
        collection_name="test_retriever_col",
        clear_existing=True
    )

    results = retrieve_top_k_chunks(index, query="attendance rules", top_k=2)
    assert len(results) <= 2
    for r in results:
        assert "text" in r
        assert "score" in r
        assert "file_name" in r
        assert "node_id" in r
        assert isinstance(r["score"], float)


def test_retrieve_empty_query_returns_empty(sample_documents, fast_embed_model, temp_chroma_dir):
    """Verify empty or whitespace query returns empty list without calling retriever."""
    nodes = split_documents_into_chunks(sample_documents)
    index = create_vector_index(
        nodes=nodes,
        embed_model=fast_embed_model,
        db_path=temp_chroma_dir,
        collection_name="test_retriever_empty",
        clear_existing=True
    )
    assert retrieve_top_k_chunks(index, "") == []
    assert retrieve_top_k_chunks(index, "   ") == []


def test_retrieve_none_index_raises():
    """Verify None index raises ValueError."""
    with pytest.raises(ValueError):
        retrieve_top_k_chunks(None, "attendance")

