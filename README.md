# 🎓 SmartCampus — Autonomous RAG Knowledge Assistant for University Rules

> **BTech Final Year Project in Computer Science & Engineering (CSE)**  
> **Title:** SmartCampus – RAG-Based Knowledge Assistant for University Rules  
> **Domain:** Academic Regulations, Examination Policies & Campus Placement Intelligence  

[![SmartCampus CI](https://github.com/jyotsnabodapati/SmartCampus/actions/workflows/ci.yml/badge.svg)](https://github.com/jyotsnabodapati/SmartCampus/actions/workflows/ci.yml)
[![Python 3.10 | 3.11 | 3.12 | 3.13](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Vector DB: ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple.svg)](https://www.trychroma.com/)
[![LLM: GPT--OSS%2020B](https://img.shields.io/badge/LLM-Groq%20Cloud%20(GPT--OSS%2020B)-orange.svg)](https://groq.com/)
[![Docker: Ready](https://img.shields.io/badge/Docker-Containerized%20(Python%203.11--slim)-blue.svg)](https://www.docker.com/)
[![Tests: 40 Passed](https://img.shields.io/badge/Pytest-40%20Passed%20(100%25)-brightgreen.svg)](https://pytest.org/)

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
- **Containerization & RAG-Ops:** Fully Dockerized application with Docker Compose, automated multi-stage CI/CD pipelines, health check smoke tests, and zero-downtime rollback procedures.

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
| **Containerization** | Docker & Compose | `3.8+` | Isolated container runtime with health checks and volume persistence |
| **Testing Suite** | Pytest | `>=7.0.0` | 40 automated unit, integration, benchmark, and smoke tests |
| **CI / Automation** | GitHub Actions | `v4 / v5` | Multi-stage pipeline: Test Matrix, Docker Build & Verify, CD Webhook |

---

## 📂 5. Project Directory Structure

```text
SmartCampus/
│
├── .github/
│   ├── workflows/
│   │   └── ci.yml                # 3-Stage CI/CD pipeline (Test Matrix, Docker Build, CD)
│   ├── pull_request_template.md  # Standardized PR checklist and review template
│   └── CODEOWNERS                # Code ownership mappings
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
├── tests/                        # Comprehensive Pytest test suite (40 automated tests)
│   ├── __init__.py               # Test package initializer
│   ├── conftest.py               # Shared test fixtures (temp dirs, mock embeddings, documents)
│   ├── test_ingestion.py         # Unit tests for document loading and metadata
│   ├── test_chunking.py          # Unit tests for sentence splitter bounds and chunk creation
│   ├── test_embeddings.py        # Unit tests for embedding generation and dimension checks
│   ├── test_vector_store.py      # Unit tests for ChromaDB storage in isolated temp folders
│   ├── test_retriever.py         # Unit tests for similarity retrieval and query bounds
│   ├── test_prompts.py           # Unit tests for prompt template construction and instructions
│   ├── test_llm.py               # Unit tests for LLM generation, Groq mock, and guardrail refusals
│   ├── test_smoke.py             # Unit tests for health-check and deployment smoke tester
│   └── test_rag_pipeline.py      # Integration tests for end-to-end RAG query flow and guardrails
│
├── Dockerfile                    # Production containerfile (Python 3.11-slim, non-root user)
├── .dockerignore                 # Excludes .env, chroma_db/, .venv/, caches from build context
├── docker-compose.yml            # Multi-container orchestration with persistent volume
├── smoke_test.py                 # Standalone health check and deployment smoke testing script
├── app.py                        # Streamlit web application with custom UI & session state
├── test_rag.py                   # 15-question benchmark evaluation script (Pytest compatible)
├── show_chroma.py                # ChromaDB inspection utility (chunks, embeddings, metadata)
├── pyrightconfig.json            # Static type analysis configuration
├── requirements.txt              # Production and development dependencies
├── .env.example                  # Environment variable configuration template
├── .gitignore                    # Comprehensive secrets, cache, and DB exclusion rules
├── ROLLBACK.md                   # Disaster recovery and rollback standard operating procedures
├── SUBMISSION.md                 # Complete RAG-Ops project assignment submission report
└── README.md                     # Complete project documentation and developer guide
```

---

## ⚙️ 6. Installation & Configuration

### Prerequisites
- Python `3.10`, `3.11`, `3.12`, or `3.13`
- Git installed on your system
- (Optional) Docker and Docker Compose
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

# Path to local ChromaDB directory (defaults to ./chroma_db)
CHROMA_DB_PATH="./chroma_db"

# Streamlit Port & Host
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS="0.0.0.0"
```

> [!NOTE]  
> If no API key is configured or internet connectivity is unavailable, SmartCampus automatically activates its built-in **MockLocalLLM** fallback engine. The application and all tests will continue to run with 100% accuracy.

---

## 🚀 7. Running the Application

### Option A: Launch Interactive Streamlit Web Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Inspect ChromaDB Storage Internals
To view stored chunks, vector dimensions, and collection metadata without starting the web UI:
```bash
python show_chroma.py
```

---

## 🐳 8. Docker Containerization & Docker Compose

SmartCampus is fully containerized using a hardened `python:3.11-slim` base image with a non-root system user (`appuser`, UID 1001) and native Streamlit health check probing.

### A. Build and Run with Docker CLI
```bash
# 1. Build the Docker image
docker build -t smartcampus:latest .

# 2. Run the container with ChromaDB volume persistence
docker run -d \
  --name smartcampus-assistant \
  -p 8501:8501 \
  -v chroma_data:/app/chroma_db \
  -e GROQ_API_KEY="your-groq-api-key" \
  smartcampus:latest

# 3. Check container logs
docker logs -f smartcampus-assistant

# 4. Probe container health
curl -f http://localhost:8501/_stcore/health
```

### B. Run with Docker Compose (Recommended)
`docker-compose.yml` mounts university documents as read-only (`:ro`) and persists ChromaDB vectors in a named volume (`chroma_data`):

```bash
# Start container in detached mode with build
docker-compose up --build -d

# Check service status and health
docker-compose ps

# View live container logs
docker-compose logs -f

# Stop container
docker-compose down
```

---

## 🧪 9. Automated Testing, Benchmarking & Smoke Testing

SmartCampus includes a three-tier automated verification system:
1. **Pytest Unit & Integration Suite:** 40 automated tests covering ingestion, chunking, embeddings, ChromaDB, retrieval, prompts, LLM mocking, and health checks.
2. **15-Question Benchmark Evaluation Suite:** Evaluates real-world university QA performance across direct, multi-chunk, and out-of-domain queries.
3. **Deployment Smoke Test:** End-to-end verification script testing documents, RAG synthesis, and Streamlit health endpoints.

### Run All Pytest Tests
```bash
pytest -v
```

### Run Benchmark Evaluation Script Directly
```bash
python test_rag.py
```

### Run Deployment Smoke Test Script
```bash
# Local testing
python smoke_test.py

# Live endpoint testing (local or deployed)
python smoke_test.py --url http://localhost:8501
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
| `tests/test_smoke.py` | `smoke_test.py` | Document presence, RAG synthesis, HTTP health checks |
| `tests/test_rag_pipeline.py` | `src/rag_engine.py` | End-to-end RAG query, source attribution, guardrails |
| `test_rag.py` | Complete System | 15 benchmark questions across Categories A, B, and C |

---

## 🔄 10. Continuous Integration & Deployment (GitHub Actions)

Continuous Integration and Deployment is orchestrated via `.github/workflows/ci.yml`.

### Pipeline Architecture:
1. **Stage 1 — Test Matrix:**
   * Tests across **Python 3.10** and **Python 3.11** on `ubuntu-latest`.
   * Runs `pytest -v --tb=short` (all 40 tests).
   * Runs `python test_rag.py` (15-question benchmark verification).
   * Runs `python smoke_test.py` (pre-build smoke check).
2. **Stage 2 — Docker Build & Health Verification:**
   * Uses `docker/setup-buildx-action` and `docker/build-push-action`.
   * Builds the Docker image inside the runner.
   * Starts the container and probes `curl -f http://localhost:8501/_stcore/health`.
   * If `DOCKERHUB_USERNAME` and `DOCKERHUB_TOKEN` are configured in GitHub Secrets, pushes:
     * `<username>/smartcampus:latest`
     * `<username>/smartcampus:<commit_sha>`
3. **Stage 3 — Continuous Deployment Trigger:**
   * Triggers automatically on pushes to `main` when all tests and Docker builds succeed.
   * Pings `DEPLOY_WEBHOOK_URL` if configured.

---

## 🌐 11. Deployment Options Analysis

For hosting the containerized Streamlit application, three production-ready options were evaluated:

| Platform | Type | Pros | Cons | Free Tier |
| :--- | :--- | :--- | :--- | :---: |
| **Hugging Face Spaces** | Managed Docker | Native Streamlit support, zero infrastructure management, automatic git push sync | Cold start times on free tier | ✅ Yes |
| **Render** | Web Service Docker | Deploy hooks support, persistent disk volumes, automatic HTTPS | Free tier spins down after 15 min of inactivity | ✅ Yes |
| **Railway / Fly.io** | Container PaaS | Sub-second deployments, custom domains, low latency | Limited trial credits | ⚠️ Limited |

**Recommended Option:** **Render** (via Docker Web Service) or **Hugging Face Spaces** (via Docker Space).

---

## 🌿 12. Git & GitHub Development Workflow

### Recommended Branch Structure

```text
main (production-ready, stable releases)
  │
  └── develop (integration branch for current sprint)
        │
        ├── feature/rag-ops-implementation
        ├── feature/enhanced-reranker
        └── bugfix/fix-pdf-table-parsing
```

* **`main`**: Production-ready code. Merged only via Pull Requests after passing CI.
* **`develop`**: Primary integration branch.
* **`feature/<name>`**: Dedicated branches for features.

---

## 🏷️ 13. Git Versioning & Release Management

SmartCampus uses semantic versioning (`vMAJOR.MINOR.PATCH`):

* **`v0.1.0`**: Initial Baseline Release (core RAG pipeline, ChromaDB indexing, Streamlit UI).
* **`v1.0.0`**: Production RAG-Ops Release (40 automated tests, Docker containerization, Docker Compose, GitHub Actions 3-stage CI/CD, health check smoke testing, and rollback recovery SOP).

### Creating and Viewing Tags
```bash
# View all annotated tags
git tag -l -n

# Create an annotated release tag
git tag -a v1.0.0 -m "Release v1.0.0: Complete RAG-Ops pipeline, Docker, and 40 automated tests"

# Push tags to GitHub
git push origin --tags
```

---

## 🔄 14. Disaster Recovery & Rollback

See [`ROLLBACK.md`](ROLLBACK.md) for complete disaster recovery standard operating procedures, covering:
* Identifying last known healthy commit SHA or tag.
* Git revert vs. hotfix branch procedures.
* Docker container rollback using commit SHA tags.
* ChromaDB vector database and document data protection.

---

## 📜 15. Submission Evidence & Deliverables

See [`SUBMISSION.md`](SUBMISSION.md) for the complete RAG-Ops assignment verification report, test outputs, and evidence checklist.
