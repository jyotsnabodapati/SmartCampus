from src.prompts import get_rag_prompt_template, RAG_SYSTEM_PROMPT_STR


def test_rag_prompt_template_placeholders():
    """Verify that the strict RAG prompt contains required variables."""
    template = get_rag_prompt_template()
    assert "{context_str}" in RAG_SYSTEM_PROMPT_STR
    assert "{query_str}" in RAG_SYSTEM_PROMPT_STR


def test_rag_prompt_anti_hallucination_instructions():
    """Verify strict anti-hallucination and refusal rules are present."""
    assert "I don't have enough information to answer that." in RAG_SYSTEM_PROMPT_STR
    assert "strictly and exclusively" in RAG_SYSTEM_PROMPT_STR.lower()


def test_rag_prompt_formatting():
    """Verify prompt formatting injects context and query accurately."""
    template = get_rag_prompt_template()
    sample_context = "[Chunk 1] Minimum attendance is 75%."
    sample_query = "What is the minimum attendance?"

    formatted = template.format(context_str=sample_context, query_str=sample_query)
    assert sample_context in formatted
    assert sample_query in formatted
    assert "Answer:" in formatted

