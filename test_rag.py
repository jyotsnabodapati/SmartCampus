import os
import sys
import warnings

# Suppress non-critical third-party warnings (huggingface hub symlinks, tensorflow logs)
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from src.rag_engine import SmartCampusRAGEngine


# 15 Benchmark Evaluation Questions
TEST_SUITE = [
    # Category A: Direct Single-Document Questions
    {
        "id": 1,
        "category": "Category A (Direct)",
        "question": "What is the minimum attendance required for exam eligibility?",
        "expected_keywords": ["75%", "75 percent"],
        "expected_sources": ["Academic_Regulations.txt"]
    },
    {
        "id": 2,
        "category": "Category A (Direct)",
        "question": "What is the fee for attendance condonation on medical grounds?",
        "expected_keywords": ["1,500", "1500", "Rs."],
        "expected_sources": ["Academic_Regulations.txt"]
    },
    {
        "id": 3,
        "category": "Category A (Direct)",
        "question": "What is the fee per subject for answer script re-evaluation?",
        "expected_keywords": ["750", "Rs."],
        "expected_sources": ["Examination_Rules.txt"]
    },
    {
        "id": 4,
        "category": "Category A (Direct)",
        "question": "What is the minimum CGPA required to participate in campus placement drives?",
        "expected_keywords": ["6.5", "CGPA"],
        "expected_sources": ["Placement_Policy.txt"]
    },
    {
        "id": 5,
        "category": "Category A (Direct)",
        "question": "What is the package cutoff for Tier 1 Dream Tier placement companies?",
        "expected_keywords": ["12", "tier 1", "lpa", "lakhs"],
        "expected_sources": ["Placement_Policy.txt"]
    },

    # Category B: Multi-Chunk / Multi-Rule Questions
    {
        "id": 6,
        "category": "Category B (Multi-Chunk)",
        "question": "What happens if a student has attendance below 65 percent?",
        "expected_keywords": ["65%", "detained", "ineligible", "repeat"],
        "expected_sources": ["Academic_Regulations.txt"]
    },
    {
        "id": 7,
        "category": "Category B (Multi-Chunk)",
        "question": "What are the rules for graduating final year students with backlogs regarding summer fast track exams?",
        "expected_keywords": ["2", "backlogs", "july", "fast-track"],
        "expected_sources": ["Examination_Rules.txt"]
    },
    {
        "id": 8,
        "category": "Category B (Multi-Chunk)",
        "question": "Can a student placed in a Tier 2 company apply for a Tier 1 company?",
        "expected_keywords": ["tier 1", "tier 2", "dream"],
        "expected_sources": ["Placement_Policy.txt"]
    },
    {
        "id": 9,
        "category": "Category B (Multi-Chunk)",
        "question": "What are the penalties for Level 2 examination malpractice?",
        "expected_keywords": ["level 2", "cancellation", "debarred", "malpractice"],
        "expected_sources": ["Examination_Rules.txt"]
    },

    {
        "id": 10,
        "category": "Category B (Multi-Chunk)",
        "question": "How many total credits are required for B.Tech degree award and what CGPA gives First Class with Distinction?",
        "expected_keywords": ["160", "8.0", "Distinction"],
        "expected_sources": ["Academic_Regulations.txt"]
    },


    # Category C: Out-of-Knowledge-Base Questions (Refusal Verification)
    {
        "id": 11,
        "category": "Category C (Out-of-Domain)",
        "question": "What is the syllabus for Quantum Physics and Machine Learning in semester 5?",
        "expected_refusal": True
    },
    {
        "id": 12,
        "category": "Category C (Out-of-Domain)",
        "question": "How many sports fields and swimming pools are available on campus?",
        "expected_refusal": True
    },
    {
        "id": 13,
        "category": "Category C (Out-of-Domain)",
        "question": "What is the hostel mess menu for Monday dinner?",
        "expected_refusal": True
    },
    {
        "id": 14,
        "category": "Category C (Out-of-Domain)",
        "question": "Who won the Indian Premier League cricket tournament in 2024?",
        "expected_refusal": True
    },
    {
        "id": 15,
        "category": "Category C (Out-of-Domain)",
        "question": "What are the salary tiers for postgraduate M.Tech students?",
        "expected_refusal": True
    }
]

def run_evaluations():
    """Runs 15-question evaluation suite against SmartCampus RAG Engine."""
    print("================================================================================")
    print("🎓 SMARTCAMPUS RAG SYSTEM AUTOMATED EVALUATION SUITE (15 BENCHMARK TEST SET)")
    print("================================================================================\n")

    engine = SmartCampusRAGEngine(data_dir="./data", db_path="./chroma_db", force_rebuild=False)

    passed_tests = 0
    refusal_tests_passed = 0
    category_c_count = 0

    print("\n| ID | Category | Question | Refusal Triggered? | Status | Sources Cited |")
    print("|---|---|---|---|---|---|")

    for test in TEST_SUITE:
        q_id = test["id"]
        cat = test["category"]
        query = test["question"]

        res = engine.query(query, top_k=3)
        ans = res["answer"]
        sources = ", ".join(res["sources"]) if res["sources"] else "None"
        is_refusal = res["is_refusal"]

        status = "FAIL"

        if test.get("expected_refusal"):
            category_c_count += 1
            if is_refusal or "don't have enough information" in ans.lower():
                status = "PASS"
                passed_tests += 1
                refusal_tests_passed += 1
        else:
            # Check keyword match or valid retrieval answer
            keywords = test.get("expected_keywords", [])
            match = any(kw.lower() in ans.lower() for kw in keywords) if keywords else True
            if match and not is_refusal:
                status = "PASS"
                passed_tests += 1

        print(f"| #{q_id:02d} | {cat} | {query[:35]}... | {'YES' if is_refusal else 'NO'} | {status} | {sources} |")

    accuracy = (passed_tests / len(TEST_SUITE)) * 100
    refusal_accuracy = (refusal_tests_passed / category_c_count * 100) if category_c_count else 100

    print("\n================================================================================")
    print("📊 FINAL EVALUATION METRICS SUMMARY")
    print("================================================================================")
    print(f" Total Benchmark Questions Evaluated : {len(TEST_SUITE)}")
    print(f" Total Passed Tests                  : {passed_tests} / {len(TEST_SUITE)}")
    print(f" System Accuracy Rate                : {accuracy:.2f}%")
    print(f" Out-of-Domain Refusal Rate          : {refusal_accuracy:.2f}% (Target: 100%)")
    print("================================================================================\n")

if __name__ == "__main__":
    run_evaluations()
