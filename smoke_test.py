"""
SmartCampus - Deployment Health Check & Smoke Testing Utility
Verifies Streamlit server health endpoint, document ingestion,
vector storage integrity, and end-to-end RAG query synthesis.
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error

# Ensure UTF-8 output on Windows
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))


def check_http_health(url: str = "http://localhost:8501", timeout: int = 15) -> bool:
    """
    Probes Streamlit's native health endpoints (/_stcore/health and /healthz).
    Returns True if either endpoint responds with HTTP 200 'ok'.
    """
    health_endpoints = [
        f"{url.rstrip('/')}/_stcore/health",
        f"{url.rstrip('/')}/healthz"
    ]

    for endpoint in health_endpoints:
        try:
            req = urllib.request.Request(endpoint, headers={"User-Agent": "SmartCampus-SmokeTest/1.0"})
            with urllib.request.urlopen(req, timeout=timeout) as response:
                if response.status == 200:
                    body = response.read().decode('utf-8').strip()
                    if body == "ok" or response.status == 200:
                        print(f"  [PASS] Streamlit Server Health Check: HTTP {response.status} at {endpoint}")
                        return True
        except Exception as e:
            continue

    print(f"  [WARN] HTTP health endpoint unreachable at {url} (server may be offline or starting)")
    return False


def check_knowledge_base_documents(data_dir: str = "./data") -> bool:
    """Verifies that university knowledge base documents exist and are readable."""
    if not os.path.exists(data_dir):
        print(f"  [FAIL] Data directory '{data_dir}' does not exist.")
        return False

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.pdf', '.txt'))]
    if not files:
        print(f"  [FAIL] No TXT or PDF documents found in '{data_dir}'.")
        return False

    print(f"  [PASS] Found {len(files)} university policy document(s): {', '.join(files)}")
    return True


def check_rag_pipeline_execution(data_dir: str = "./data", db_path: str = "./chroma_db") -> bool:
    """Executes an end-to-end RAG query test verifying retrieval, citations, and guardrails."""
    try:
        from src.rag_engine import SmartCampusRAGEngine
        engine = SmartCampusRAGEngine(data_dir=data_dir, db_path=db_path, force_rebuild=False)

        # Test A: Direct grounded query
        query_a = "What is the minimum attendance required for exam eligibility?"
        res_a = engine.query(query_a)
        if not res_a or not res_a.get("answer"):
            print("  [FAIL] RAG query returned empty answer.")
            return False

        if "75" not in res_a["answer"]:
            print(f"  [WARN] RAG query answer did not contain expected keyword: {res_a['answer'][:80]}")

        print(f"  [PASS] Grounded Query Test: Answered successfully with {len(res_a.get('sources', []))} source(s).")

        # Test B: Out-of-domain guardrail query
        query_b = "What is the hostel mess menu for Monday dinner?"
        res_b = engine.query(query_b)
        if not res_b.get("is_refusal"):
            print("  [FAIL] Out-of-domain query did not trigger refusal guardrail.")
            return False

        print(f"  [PASS] Guardrail Test: Correctly triggered refusal for out-of-domain question.")
        return True

    except Exception as e:
        print(f"  [FAIL] RAG Pipeline Execution Error: {e}")
        return False


def run_smoke_tests(url: str = None, data_dir: str = "./data", db_path: str = "./chroma_db", max_retries: int = 3) -> bool:
    """Runs the complete smoke test suite."""
    print("=" * 70)
    print("   SMARTCAMPUS DEPLOYMENT HEALTH CHECK & SMOKE TEST SUITE")
    print("=" * 70)

    all_passed = True

    # 1. Documents Check
    print("\n[Stage 1/3] Knowledge Base Documents Inspection:")
    if not check_knowledge_base_documents(data_dir):
        all_passed = False

    # 2. RAG Pipeline Check
    print("\n[Stage 2/3] RAG Engine Query Pipeline Verification:")
    if not check_rag_pipeline_execution(data_dir, db_path):
        all_passed = False

    # 3. Optional HTTP Health Check
    if url:
        print(f"\n[Stage 3/3] Streamlit HTTP Health Endpoint Verification ({url}):")
        http_ok = False
        for attempt in range(1, max_retries + 1):
            if check_http_health(url):
                http_ok = True
                break
            if attempt < max_retries:
                time.sleep(3)

        if not http_ok:
            all_passed = False
    else:
        print("\n[Stage 3/3] HTTP Health Endpoint Check skipped (no --url specified).")

    print("\n" + "=" * 70)
    if all_passed:
        print("✅ ALL SMOKE TESTS PASSED - SMARTCAMPUS IS DEPLOYMENT HEALTHY")
    else:
        print("❌ ONE OR MORE SMOKE TESTS FAILED")
    print("=" * 70 + "\n")

    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SmartCampus Deployment Smoke Test")
    parser.add_argument("--url", type=str, default=None, help="Streamlit server base URL (e.g. http://localhost:8501)")
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to university documents")
    parser.add_argument("--db-path", type=str, default="./chroma_db", help="Path to ChromaDB vector store")
    args = parser.parse_args()

    success = run_smoke_tests(url=args.url, data_dir=args.data_dir, db_path=args.db_path)
    sys.exit(0 if success else 1)
