import os
from unittest.mock import patch
from src.llm import get_llm, MockLocalLLM


def test_get_llm_without_api_key(monkeypatch):
    """Verify fallback to MockLocalLLM when no GROQ_API_KEY is present."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    llm = get_llm()
    assert isinstance(llm, MockLocalLLM)


def test_mock_llm_direct_academic_answer(mock_llm):
    """Verify grounded answer generation for attendance regulations."""
    prompt = "User Question: What is the minimum attendance required for exam eligibility?"
    response = mock_llm.complete(prompt)
    assert "75%" in response.text
    assert "Academic Regulations" in response.text


def test_mock_llm_exam_reevaluation_answer(mock_llm):
    """Verify grounded answer generation for re-evaluation policy."""
    prompt = "User Question: What is the fee for re-evaluation of answer scripts?"
    response = mock_llm.complete(prompt)
    assert "750" in response.text
    assert "Examination Rules" in response.text


def test_mock_llm_placement_policy_answer(mock_llm):
    """Verify grounded answer generation for placement policy."""
    prompt = "User Question: What is the cutoff for Tier 1 Dream Tier placements?"
    response = mock_llm.complete(prompt)
    assert "12 LPA" in response.text
    assert "Placement Policy" in response.text


def test_mock_llm_out_of_domain_refusal(mock_llm):
    """Verify strict refusal for questions outside the university document context."""
    out_of_domain_queries = [
        "User Question: What is the syllabus for Quantum Physics?",
        "User Question: How many swimming pools are on campus?",
        "User Question: What is the hostel mess menu for dinner?",
        "User Question: Who won the IPL cricket tournament?"
    ]
    expected_refusal = "I don't have enough information to answer that."
    for q in out_of_domain_queries:
        res = mock_llm.complete(q)
        assert expected_refusal in res.text, f"Failed refusal for query: {q}"


def test_mock_llm_casual_and_personal_refusal(mock_llm):
    """Verify refusal for casual greetings and personal questions."""
    casual_queries = [
        "User Question: hello",
        "User Question: what is my name",
        "User Question: who are you"
    ]
    expected_refusal = "I don't have enough information to answer that."
    for q in casual_queries:
        res = mock_llm.complete(q)
        assert expected_refusal in res.text


def test_mock_groq_initialization(monkeypatch):
    """Verify Groq initialization with mocked Groq class."""
    monkeypatch.setenv("GROQ_API_KEY", "gsk_test_mock_key_12345")
    with patch("src.llm.Groq") as mock_groq_class:
        mock_groq_class.return_value = "MockGroqInstance"
        llm = get_llm(model_name="openai/gpt-oss-20b")
        assert llm == "MockGroqInstance"
        mock_groq_class.assert_called_once_with(
            model="openai/gpt-oss-20b",
            api_key="gsk_test_mock_key_12345",
            temperature=0.1
        )

