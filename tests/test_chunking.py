import pytest
from src.chunking import get_sentence_splitter, split_documents_into_chunks


def test_get_sentence_splitter_defaults():
    """Verify default parameters of sentence splitter."""
    splitter = get_sentence_splitter()
    assert splitter.chunk_size == 512
    assert splitter.chunk_overlap == 64


def test_get_sentence_splitter_custom():
    """Verify custom parameters of sentence splitter."""
    splitter = get_sentence_splitter(chunk_size=256, chunk_overlap=32)
    assert splitter.chunk_size == 256
    assert splitter.chunk_overlap == 32


def test_get_sentence_splitter_invalid_parameters():
    """Verify parameter bounds validation."""
    with pytest.raises(ValueError):
        get_sentence_splitter(chunk_size=0)
    with pytest.raises(ValueError):
        get_sentence_splitter(chunk_size=100, chunk_overlap=-5)
    with pytest.raises(ValueError):
        get_sentence_splitter(chunk_size=100, chunk_overlap=100)


def test_split_documents_into_chunks(sample_documents):
    """Verify document chunking and node metadata propagation."""
    nodes = split_documents_into_chunks(sample_documents, chunk_size=256, chunk_overlap=32)
    assert len(nodes) >= len(sample_documents)

    # Check node metadata and content
    for node in nodes:
        assert node.get_content()
        assert "file_name" in node.metadata
        assert node.metadata["file_name"] in [
            "Academic_Regulations.txt",
            "Examination_Rules.txt",
            "Placement_Policy.txt"
        ]


def test_split_empty_documents():
    """Verify splitting empty document list gracefully returns empty list."""
    nodes = split_documents_into_chunks([])
    assert nodes == []

