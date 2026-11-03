import pytest
from src.chunking import split_documents_into_chunks
from src.vector_store import (
    get_chroma_vector_store,
    create_vector_index,
    load_existing_index
)


def test_get_chroma_vector_store(temp_chroma_dir):
    """Verify initializing a persistent ChromaDB vector store in an isolated test folder."""
    vector_store = get_chroma_vector_store(
        db_path=temp_chroma_dir,
        collection_name="test_collection",
        clear_existing=True
    )
    assert vector_store is not None


def test_create_and_load_vector_index(sample_documents, fast_embed_model, temp_chroma_dir):
    """
    Verify creating an index and loading it back from ChromaDB without touching production db.
    """
    nodes = split_documents_into_chunks(sample_documents)
    
    # 1. Create index
    index = create_vector_index(
        nodes=nodes,
        embed_model=fast_embed_model,
        db_path=temp_chroma_dir,
        collection_name="test_collection",
        clear_existing=True
    )
    assert index is not None

    # 2. Load existing index
    loaded_index = load_existing_index(
        embed_model=fast_embed_model,
        db_path=temp_chroma_dir,
        collection_name="test_collection"
    )
    assert loaded_index is not None


def test_create_vector_index_empty_nodes_raises(fast_embed_model, temp_chroma_dir):
    """Verify that attempting to index an empty nodes list raises ValueError."""
    with pytest.raises(ValueError):
        create_vector_index(
            nodes=[],
            embed_model=fast_embed_model,
            db_path=temp_chroma_dir,
            collection_name="test_empty"
        )


def test_load_non_existent_collection_returns_none(fast_embed_model, temp_chroma_dir):
    """Verify loading a non-existent collection returns None."""
    loaded = load_existing_index(
        embed_model=fast_embed_model,
        db_path=temp_chroma_dir,
        collection_name="non_existent_collection"
    )
    assert loaded is None

