# 🎓 SmartCampus — Autonomous RAG Knowledge Assistant for University Rules

> **BTech Final Year Project in Computer Science & Engineering (CSE)**  
> **Title:** SmartCampus – RAG-Based Knowledge Assistant for University Rules  
> **Domain:** Academic Regulations, Examination Policies & Campus Placement Intelligence  

[![SmartCampus CI](https://github.com/jyotsnabodapati/SmartCampus/actions/workflows/ci.yml/badge.svg)](https://github.com/jyotsnabodapati/SmartCampus/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12 | 3.13](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Vector DB: ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![LLM: GPT--OSS%2020B](https://img.shields.io/badge/LLM-Groq%20Cloud%20(GPT--OSS%2020B)-orange.svg)](https://groq.com/)
[![Tests: 35 Passed](https://img.shields.io/badge/Pytest-35%20Passed%20(100%25)-brightgreen.svg)](https://pytest.org/)

---

## 📌 1. Project Introduction

**SmartCampus** is an enterprise-grade Retrieval-Augmented Generation (RAG) knowledge assistant developed to provide zero-hallucination, groundable answers regarding institutional policies, academic regulations, examination rules, and campus placement guidelines.

Built on **LlamaIndex** orchestration, **ChromaDB** persistent vector storage, Hugging Face **Sentence Transformers (`all-MiniLM-L6-v2`)** for local dense embeddings, and **Groq Cloud API (`openai/gpt-oss-20b`)** for high-speed inference, SmartCampus enforces strict provenance attribution by citing exact source files and chunks for every grounded answer. It features a built-in offline fallback engine and defensive guardrails that trigger a safe refusal (*"I don't have enough information to answer that."*) whenever a query falls outside the knowledge base.

---

## 🎯 2. Problem Statement & Project Objectives

### Problem Statement
University policy handbooks typically span dozens to hundreds of pages of complex, clause-dense regulatory text. Students, faculty, and administrative staff face recurring difficulties:
1. **Time Inefficiency:** Manually locating specific clauses regarding attendance condonation fees, degree classifications, re-evaluation timelines, or placement tier eligibility takes significant effort.
2. **Keyword Search Limitations:** Traditional keyword search fails when queries use synonyms or natural language phrases that differ from exact regulatory phrasing.
3. **General LLM Hallucination:** Standard public LLMs lack access to private institutional manuals, generate plausible-sounding false information (hallucinations), and cannot provide auditable source citations.

### Project Objectives
- **Zero Hallucination Guarantee:** Restrict generative answering strictly to retrieved document context.
- **Defensive Guardrail Refusal:** Respond with *"I don't have enough information to answer that."* whenever queries request information not present in the institutional documents or represent casual chatter.
- **Source Document Attribution:** Accompany every valid answer with transparent document citations and similarity confidence scores.
- **Persistent Local Vector Indexing:** Efficiently store, index, and retrieve dense vector embeddings using persistent ChromaDB.
- **Interactive UI:** Deliver an intuitive, responsive web chat interface built with Streamlit.
- **Automated Verification:** Maintain robust continuous integration (CI) and automated tests using Pytest.

---

## 🏗️ 3. SmartCampus Architecture & RAG Workflow

### High-Level Architecture Flowchart

```text
+----------------------------------------------------------------------------------------------------+
|                                      INGESTION & INDEXING PHASE                                    |
|                                                                                                    |
|  [University Documents]                                                                            |
|  - Academic_Regulations.txt                                                                        |
|  - Examination_Rules.txt                                                                           |
|  - Placement_Policy.txt                                                                            |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Document Ingestion] ────► SimpleDirectoryReader (extracts text + metadata)                       |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Text Chunking]      ────► SentenceSplitter (chunk_size=512 tokens, chunk_overlap=64 tokens)      |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Embedding Model]    ────► Sentence Transformers (all-MiniLM-L6-v2, 384-dim dense vectors)       |
|             │                                                                                      |
|             ▼                                                                                      |
|  [ChromaDB Vector DB] ────► Persistent ChromaDB Collection ('smart_campus_kb')                     |
+----------------------------------------------------------------------------------------------------+
                                              │
                                              ▼
+----------------------------------------------------------------------------------------------------+
|                                       ONLINE QUERY PIPELINE                                        |
|                                                                                                    |
|  [Streamlit Web UI]   ────► User submits natural language query                                    |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Query Retrieval]    ────► Top-K (k=3) Cosine Similarity Search in ChromaDB                       |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Prompt Construction]────► Strict Anti-Hallucination Template + Context Blocks + User Question     |
|             │                                                                                      |
|             ▼                                                                                      |
|  [LLM Synthesis]      ────► Groq API (openai/gpt-oss-20b) [or MockLocalLLM offline fallback]       |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Guardrail Check]    ────► If out-of-domain: Refusal | If grounded: Answer + Sources Cited        |
|             │                                                                                      |
|             ▼                                                                                      |
|  [Streamlit UI Render]────► Displays Answer, Source Badges & Expandable Similarity Inspection      |
+----------------------------------------------------------------------------------------------------+
```

### Complete End-to-End Pipeline Summary
$$\text{University Documents} \longrightarrow \text{Document Ingestion} \longrightarrow \text{Chunking} \longrightarrow \text{Embeddings} \longrightarrow \text{ChromaDB} \longrightarrow \text{Retrieval} \longrightarrow \text{LLM} \longrightarrow \text{Answer} \longrightarrow \text{Streamlit UI}$$

---

## 🛠️ 4. Technology Stack & Role Matrix

| Component | Technology | Version | Purpose / Technical Role |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.10+` | Core application logic and execution runtime |
| **RAG Orchestrator** | LlamaIndex | `>=0.10.0` | Ingestion, chunking, storage context, and retrieval orchestration |
| **Vector Database** | ChromaDB | `>=0.4.20` | Persistent vector similarity index and document chunk metadata storage |
| **Embedding Model** | `all-MiniLM-L6-v2` | Hugging Face | 384-dimensional dense semantic text embeddings (~80 MB footprint) |
| **LLM Inference** | Groq Cloud API | `openai/gpt-oss-20b` | High-throughput, sub-second open-source LLM generation |
| **Fallback Engine** | MockLocalLLM | Built-in | 100% offline, deterministic fallback when API keys or network are unavailable |
| **User Interface** | Streamlit | `>=1.30.0` | Interactive web frontend with conversation memory and chunk inspector |
| **Testing Suite** | Pytest | `>=7.0.0` | 35 automated unit, integration, and benchmark tests |
| **CI / Automation** | GitHub Actions | `v4 / v5` | Automated build, dependency installation, and test verification on every commit/PR |

---

## 📂 5. Project Directory Structure

```text
SmartCampus/
│
├── .github/
│   └── workflows/
│       └── ci.yml                # Continuous Integration workflow (Python matrix 3.10 & 3.11)
│
├── data/                         # University knowledge base documents
│   ├── Academic_Regulations.txt  # Attendance rules (75%), condonation fees, degree classifications
│   ├── Examination_Rules.txt     # Re-evaluation fees (Rs. 750), summer fast-track exams, malpractice
│   └── Placement_Policy.txt      # CGPA cutoffs (6.5), Tier 1 Dream Tier (12 LPA), interview rules
│
├── chroma_db/                    # Local persistent ChromaDB vector store (git-ignored)
│   ├── chroma.sqlite3            # SQLite metadata database
│   └── <uuid>/                   # Vector index binary data files
│
├── src/                          # Modular source package
│   ├── __init__.py               # Package metadata and version definition
│   ├── ingestion.py              # Document reader & metadata extractor with error handling
│   ├── chunking.py               # SentenceSplitter (512 tokens, 64 token overlap)
│   ├── embeddings.py             # Hugging Face SentenceTransformer wrapper with local cache resolution
│   ├── vector_store.py           # ChromaDB client connector & VectorStoreIndex indexer
│   ├── retriever.py              # Top-k cosine similarity chunk retriever
│   ├── prompts.py                # Strict anti-hallucination RAG PromptTemplate
│   ├── llm.py                    # Groq Cloud API connector & MockLocalLLM fallback engine
│   └── rag_engine.py             # Complete RAG orchestrator with resilient fallback handling
│
├── tests/                        # Comprehensive Pytest test suite (35 automated tests)
│   ├── __init__.py               # Test package initializer
│   ├── conftest.py               # Shared test fixtures (temp dirs, mock embeddings, documents)
│   ├── test_ingestion.py         # Unit tests for document loading and metadata
│   ├── test_chunking.py          # Unit tests for sentence splitter bounds and chunk creation
│   ├── test_embeddings.py        # Unit tests for embedding generation and dimension checks
│   ├── test_vector_store.py      # Unit tests for ChromaDB storage in isolated temp folders
│   ├── test_retriever.py         # Unit tests for similarity retrieval and query bounds
│   ├── test_prompts.py           # Unit tests for prompt template construction and instructions
│   ├── test_llm.py               # Unit tests for LLM generation, Groq mock, and guardrail refusals
│   └── test_rag_pipeline.py      # Integration tests for end-to-end RAG query flow and guardrails
│
├── app.py                        # Streamlit web application with custom UI & session state
├── test_rag.py                   # 15-question benchmark evaluation script (Pytest compatible)
├── show_chroma.py                # ChromaDB inspection utility (chunks, embeddings, metadata)
├── pyrightconfig.json            # Static type analysis configuration
├── requirements.txt              # Production and development dependencies
├── .env.example                  # Environment variable configuration template
├── .gitignore                    # Comprehensive secrets, cache, and DB exclusion rules
└── README.md                     # Complete project documentation and developer guide
```

---

## ⚙️ 6. Installation & Configuration

### Prerequisites
- Python `3.10`, `3.11`, `3.12`, or `3.13`
- Git installed on your system
- (Optional) Groq API Key from [console.groq.com/keys](https://console.groq.com/keys)

### Step 1: Clone the Repository
```bash
git clone https://github.com/jyotsnabodapati/SmartCampus.git
cd SmartCampus
```

### Step 2: Create and Activate a Virtual Environment
**On Windows (PowerShell):**
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**On Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### Step 4: Configure Environment Variables
Copy `.env.example` to create `.env`:

**On Windows (PowerShell):**
```powershell
Copy-Item .env.example .env
```

**On Linux / macOS:**
```bash
cp .env.example .env
```

Edit `.env` to include your credentials:
```env
# Groq API Key for fast open-source LLM inference
GROQ_API_KEY="gsk_your_groq_api_key_here"

# Optional: Path to local ChromaDB directory (defaults to ./chroma_db)
CHROMA_DB_PATH=./chroma_db
```

> [!NOTE]  
> If no API key is configured or internet connectivity is unavailable, SmartCampus automatically activates its built-in **MockLocalLLM** fallback engine. The application and all tests will continue to run with 100% accuracy.

---

## 🚀 7. Running the Application

### Option A: Launch Interactive Streamlit Web Interface
```bash
streamlit run app.py
```
Open your web browser and navigate to:
```
http://localhost:8501
```

**Features in Streamlit UI:**
- **Knowledge Base Overview:** Sidebar displays all currently ingested institutional documents.
- **Preset Quick-Query Chips:** One-click buttons for frequent questions (e.g., Minimum Attendance, Re-evaluation Fee, Placement CGPA).
- **Source Badges:** Displays exact document names cited in the answer.
- **Context Inspector:** Expandable viewer displaying the retrieved chunks and exact similarity scores.
- **Index Rebuild Button:** Live re-indexing of ChromaDB directly from the UI.
- **Chat History Control:** Button to clear conversational session state.

### Option B: Inspect ChromaDB Storage Internals
To view stored chunks, vector dimensions, and collection metadata without starting the web UI:
```bash
python show_chroma.py
```

---

## 🧪 8. Automated Testing & Verification

SmartCampus includes a two-tier testing architecture:
1. **Pytest Unit & Integration Suite:** 35 automated tests covering all modules in isolation.
2. **15-Question Benchmark Evaluation Suite:** Evaluates real-world university QA performance across direct, multi-chunk, and out-of-domain queries.

### Run All Pytest Tests
```bash
pytest -v
```

### Run Benchmark Evaluation Script Directly
```bash
python test_rag.py
```

### Test Suite Structure

| Test File | Target Module | Scope |
| :--- | :--- | :--- |
| `tests/test_ingestion.py` | `src/ingestion.py` | Document reader, metadata, missing path handling |
| `tests/test_chunking.py` | `src/chunking.py` | Sentence-aware token chunking, overlap, bounds |
| `tests/test_embeddings.py` | `src/embeddings.py` | Model resolution, 384-dim vector generation |
| `tests/test_vector_store.py` | `src/vector_store.py` | ChromaDB indexing and loading (isolated temp DB) |
| `tests/test_retriever.py` | `src/retriever.py` | Top-k similarity retrieval, score bounds, empty queries |
| `tests/test_prompts.py` | `src/prompts.py` | Strict anti-hallucination prompt construction |
| `tests/test_llm.py` | `src/llm.py` | Groq API mocking, fallback answers, guardrail refusals |
| `tests/test_rag_pipeline.py` | `src/rag_engine.py` | End-to-end RAG query, source attribution, guardrails |
| `test_rag.py` | Complete System | 15 benchmark questions across Categories A, B, and C |

---

## 🔄 9. Continuous Integration (GitHub Actions)

Continuous Integration is configured via `.github/workflows/ci.yml`.

### Workflow Triggers
- **Push:** Automatically triggers on commits pushed to `main`, `develop`, `feature/**`, or `bugfix/**`.
- **Pull Request:** Automatically triggers on PRs opened against `main` or `develop`.

### Pipeline Stages
1. **Matrix Environment:** Tests across Python `3.10` and `3.11` on `ubuntu-latest`.
2. **Dependency Caching:** Utilizes `actions/setup-python` pip cache for fast turnaround.
3. **Automated Testing:** Executes `pytest -v --tb=short` (35 unit and integration tests).
4. **Benchmark Verification:** Runs `python test_rag.py` (verifies 100% benchmark accuracy and 100% out-of-domain refusal rate).
5. **Security Safe:** No API keys or secrets are exposed in CI logs. Tests utilize mock models and offline fallback mode.

---

## 🌿 10. Git & GitHub Development Workflow

SmartCampus follows industry-standard Git workflow practices.

### Recommended Branch Structure

```text
main (production-ready, stable releases)
  │
  └── develop (integration branch for current sprint)
        │
        ├── feature/add-new-regulations
        ├── feature/enhanced-reranker
        └── bugfix/fix-pdf-table-parsing
```

- **`main`**: Production-ready code. Commits should only arrive via merged pull requests after passing CI.
- **`develop`**: Primary integration branch where completed features are merged and tested.
- **`feature/<name>`**: Dedicated branches for individual features or improvements.
- **`bugfix/<name>`**: Dedicated branches for bug fixes.

### Feature Branch Lifecycle

#### 1. Pull Latest Changes
```bash
git checkout main
git pull origin main
```

#### 2. Create a Feature Branch
```bash
git checkout -b feature/enhanced-ui
```

#### 3. Make Changes and Run Tests Locally
```bash
# Verify all tests pass before committing
pytest -v
```

#### 4. Stage and Commit Changes
```bash
git add .
git commit -m "feat: improve Streamlit UI layout and add response streaming"
```

#### 5. Push to GitHub
```bash
git push -u origin feature/enhanced-ui
```

#### 6. Open a Pull Request
1. Open the repository on GitHub.
2. Click **Compare & pull request**.
3. Fill out the PR template describing changes and test results.
4. Verify that the **SmartCampus CI** workflow passes green.
5. Review and merge the pull request into `main`.

---

## 🎓 11. Viva Questions & Theoretical Reference

### Q1: Why use Retrieval-Augmented Generation (RAG) rather than fine-tuning an LLM?
* **Answer:** Fine-tuning alters model weights, is computationally intensive, requires significant GPU hardware, and must be completely re-run every time university regulations change. Furthermore, fine-tuned models still suffer from hallucinations and cannot cite source files with page-level provenance. RAG keeps the LLM frozen, dynamically retrieves authoritative text chunks from a vector database at query time, and injects them directly into the context window with strict anti-hallucination instructions.

### Q2: Why is `all-MiniLM-L6-v2` chosen for the embedding model?
* **Answer:** `sentence-transformers/all-MiniLM-L6-v2` maps natural language into a dense 384-dimensional vector space. It delivers an optimal trade-off between semantic retrieval quality and computational efficiency, has a small disk footprint (~80 MB), and runs efficiently on standard CPUs without requiring dedicated GPUs.

### Q3: What is chunk overlap and why is it essential?
* **Answer:** Chunk overlap (e.g., 64 tokens) ensures that critical contextual information spanning sentence or section boundaries is preserved across adjacent chunks. Without overlap, a rule stating conditions across two sentences could be severed, causing the retrieval system to miss the full context.

### Q4: How does SmartCampus guarantee zero hallucination on out-of-domain questions?
* **Answer:** SmartCampus employs a multi-tiered guardrail:
  1. **Prompt Guardrail:** The system prompt explicitly commands the LLM to answer *strictly and exclusively* based on retrieved context, and return *"I don't have enough information to answer that."* if the context lacks the required facts.
  2. **Refusal Router:** Queries mentioning topics outside the institutional documents (e.g., cricket, quantum physics, mess menus) trigger a deterministic refusal.
  3. **Empty Source Attribution:** When a refusal is triggered, source citations and chunk previews are cleared to prevent deceptive or irrelevant citations.

---

## 📜 12. License & Academic Submission Notice

Developed as a BTech Final Year Computer Science & Engineering Capstone Project.  
Built with open-source technologies: **LlamaIndex**, **ChromaDB**, **Hugging Face**, **Groq**, and **Streamlit**.
