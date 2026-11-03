import pytest
from unittest.mock import patch, MagicMock
from smoke_test import check_knowledge_base_documents, check_rag_pipeline_execution, check_http_health


def test_smoke_documents_exist():
    """Verify smoke test passes on the actual project ./data directory."""
    assert check_knowledge_base_documents("./data") is True


def test_smoke_documents_missing():
    """Verify smoke test catches missing data directory."""
    assert check_knowledge_base_documents("./non_existent_data_directory_xyz") is False


def test_smoke_rag_pipeline():
    """Verify smoke test executes RAG pipeline query and guardrails successfully."""
    assert check_rag_pipeline_execution("./data", "./chroma_db") is True


def test_smoke_http_health_success():
    """Verify HTTP health checker handles 200 'ok' response correctly."""
    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.read.return_value = b"ok"
    mock_resp.__enter__.return_value = mock_resp

    with patch("urllib.request.urlopen", return_value=mock_resp):
        assert check_http_health("http://localhost:8501") is True


def test_smoke_http_health_unreachable():
    """Verify HTTP health checker fails gracefully when endpoint is unreachable."""
    with patch("urllib.request.urlopen", side_effect=Exception("Connection refused")):
        assert check_http_health("http://invalid-localhost-url:9999", timeout=1) is False
