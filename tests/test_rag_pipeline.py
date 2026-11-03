import pytest
from src.rag_engine import SmartCampusRAGEngine


@pytest.fixture(scope="module")
def rag_engine_instance():
    """Provides a shared RAG Engine instance using existing ChromaDB index."""
    return SmartCampusRAGEngine(data_dir="./data", db_path="./chroma_db", force_rebuild=False)


def test_rag_query_valid_academic_question(rag_engine_instance):
    """Verify end-to-end RAG answer for academic attendance query."""
    result = rag_engine_instance.query("What is the minimum attendance required for exam eligibility?")
    assert "question" in result
    assert "answer" in result
    assert "sources" in result
    assert "retrieved_chunks" in result
    assert "is_refusal" in result

    assert not result["is_refusal"]
    assert "75%" in result["answer"] or "75 percent" in result["answer"].lower()
    assert any("Academic_Regulations" in s for s in result["sources"])
    assert len(result["retrieved_chunks"]) > 0


def test_rag_query_empty_input(rag_engine_instance):
    """Verify graceful handling of empty or whitespace queries."""
    result = rag_engine_instance.query("   ")
    assert result["is_refusal"]
    assert "Please ask a question" in result["answer"]
    assert result["sources"] == []


def test_rag_query_out_of_domain_refusal(rag_engine_instance):
    """Verify strict guardrail refusal for queries outside university policy documents."""
    result = rag_engine_instance.query("What is the hostel mess menu for Monday dinner?")
    assert result["is_refusal"]
    assert "I don't have enough information to answer that." in result["answer"]
    assert result["sources"] == []
    assert result["retrieved_chunks"] == []


def test_rag_query_malpractice_policy(rag_engine_instance):
    """Verify multi-rule query returns grounded answer with Examination Rules source citation."""
    result = rag_engine_instance.query("What are the penalties for Level 2 examination malpractice?")
    assert not result["is_refusal"]
    assert any("Examination_Rules" in s for s in result["sources"])
    assert len(result["retrieved_chunks"]) >= 1


def test_rag_query_placement_policy(rag_engine_instance):
    """Verify placement policy query returns grounded answer with Placement Policy source citation."""
    result = rag_engine_instance.query("What is the minimum CGPA required to participate in campus placement drives?")
    assert not result["is_refusal"]
    assert any("Placement_Policy" in s for s in result["sources"])
    assert "6.5" in result["answer"]
