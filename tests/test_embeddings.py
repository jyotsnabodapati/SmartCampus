import pytest
from src.embeddings import get_embedding_model, resolve_model_path


def test_resolve_model_path():
    """Verify that model path resolver returns a valid string."""
    resolved = resolve_model_path("sentence-transformers/all-MiniLM-L6-v2")
    assert isinstance(resolved, str)
    assert len(resolved) > 0


def test_fast_mock_embedding(fast_embed_model):
    """Verify mock embedding model produces correct 384-dimensional vectors."""
    text = "Minimum attendance required is 75 percent."
    vector = fast_embed_model.get_text_embedding(text)
    assert len(vector) == 384
    assert isinstance(vector[0], float)


def test_embedding_model_generation():
    """
    Verify the production Hugging Face embedding model generates 384-dimensional embeddings.
    Uses the configured sentence-transformers/all-MiniLM-L6-v2 model.
    """
    model = get_embedding_model()
    assert model is not None
    sample_text = "SmartCampus university rules test."
    vector = model.get_text_embedding(sample_text)
    assert len(vector) == 384
    assert isinstance(vector, list)

