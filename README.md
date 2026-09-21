# 🎓 SmartCampus — Autonomous RAG Knowledge Assistant

> **BTech Final Year Project in Computer Science & Engineering (CSE)**  
> **Title:** Building a Retrieval-Augmented Generation (RAG) Application with Open-Source LLMs  
> **Domain:** Academic Regulations & Campus Knowledge Intelligence  

---

## 📌 Project Overview

**SmartCampus** is an enterprise-grade Knowledge Assistant chatbot designed to eliminate hallucination in domain-specific QA. Built using **Retrieval-Augmented Generation (RAG)**, it ingests official institutional documents (PDF/TXT), parses and splits them into semantic chunks, indexes them into a persistent vector database (**ChromaDB**) using local Hugging Face embeddings (`all-MiniLM-L6-v2`), and leverages open-source Large Language Models (**Llama 3.1 8B via Groq API**) to generate precise, grounded answers accompanied by exact source document citations.

---

## 🎯 Problem Statement & Objectives

### Problem Statement
Students, faculty, and administrative staff spend considerable time manually sifting through dense 50+ page university policy manuals to answer specific questions regarding attendance, re-evaluation fees, grading criteria, and placement eligibility. Traditional keyword searching fails due to semantic variation in natural language queries, while standard LLMs suffer from **hallucination** and lack access to private institutional policies.

### Core Objectives
1. **Zero Hallucination:** Constrain LLM responses strictly to retrieved document context.
2. **Fallback Refusal:** Respond with *"I don't have enough information to answer that."* when query context is absent from the knowledge base.
3. **Source Transparency:** Attribute every answer to its original document source file and provide similarity scores.
4. **Local Vector Storage:** Store and query high-dimensional embeddings using persistent **ChromaDB**.
5. **Interactive UI:** Provide a clean, browser-based **Streamlit** chat interface.

---

## 🏗️ System Architecture

```text
 Documents (PDF/TXT)
       │
       ▼
 Document Reader (LlamaIndex)
       │
       ▼
 Sentence-Aware Chunking (chunk_size=512, overlap=64)
       │
       ▼
 Hugging Face Embedding Generator (all-MiniLM-L6-v2, 384-dim)
       │
       ▼
 Persistent ChromaDB Vector Store
       │
 [Online Query Pipeline]
       │
 User Query ──► Query Embedding ──► Cosine Similarity Search (Top-K=3)
                                             │
                                             ▼
 LLM Prompt Construction (Strict Prompt + Retrieved Context)
                                             │
                                             ▼
                         Open-Source LLM (Groq Llama 3.1)
                                             │
                                             ▼
                       Grounded Answer + Source Citations
```

---

## 🛠️ Technology Stack & Role Matrix

| Component | Technology | Version | Purpose / Role |
| :--- | :--- | :--- | :--- |
| **Language** | Python | `3.10+` | Core application development |
| **RAG Framework** | LlamaIndex | `>=0.10.0` | Ingestion, node parsing, retriever & prompt orchestration |
| **Embedding Model** | `all-MiniLM-L6-v2` | Hugging Face | 384-dimensional vector embedding generation |
| **Vector DB** | ChromaDB | `>=0.4.20` | Persistent vector similarity index |
| **LLM Provider** | Groq API | Llama 3.1 8B | Sub-second open-source LLM inference |
| **Web Interface** | Streamlit | `>=1.30.0` | Browser-based interactive chat frontend |
| **Testing** | Custom Suite | Python | 15-question benchmark evaluation script |

---

## 📂 Directory Structure

```text
rag_knowledge_assistant/
│
├── data/                         # Knowledge base raw document directory
│   ├── Academic_Regulations.txt  # Attendance rules, grading scale, credit limits
│   ├── Examination_Rules.txt     # Re-evaluation fees, fast-track summer exams, malpractice
│   └── Placement_Policy.txt      # CGPA cutoff, Tier 1 dream tier, backlog rules
│
├── chroma_db/                    # Local persistent ChromaDB vector store
│
├── src/                          # Modular source code
│   ├── __init__.py               # Package initializer
│   ├── ingestion.py              # Document reader & metadata extractor
│   ├── chunking.py               # SentenceSplitter (512 tokens, 64 overlap)
│   ├── embeddings.py             # Hugging Face SentenceTransformer model wrapper
│   ├── vector_store.py           # ChromaDB client & collection indexer
│   ├── retriever.py              # Standalone top-k similarity retriever
│   ├── llm.py                    # Groq API provider & local fallback LLM
│   ├── prompts.py                # Strict anti-hallucination prompt template
│   └── rag_engine.py             # Complete RAG pipeline query engine
│
├── app.py                        # Streamlit web application frontend
├── test_rag.py                   # Automated 15-question test evaluator script
├── requirements.txt              # Project dependencies list
├── .env.example                  # API Key environment variable template
└── README.md                     # Project documentation & viva guide
```

---

## 🚀 Quickstart & Setup Guide

### Step 1: Clone or Open Project Workspace
```bash
cd "c:\Users\jyots\Desktop\AI Spartans"
```

### Step 2: Install Required Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Configure Environment Variables
Copy `.env.example` to `.env` and set your Groq API key:
```bash
cp .env.example .env
```
Edit `.env`:
```env
GROQ_API_KEY=gsk_your_groq_api_key_here
```
*(Note: If no API key is provided, the application automatically uses a local fallback mode so you can test the pipeline without interruptions).*

---

## 🧪 Running the System

### Option A: Launch Interactive Streamlit Web Interface
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### Option B: Run Automated 15-Question Benchmark Test Suite
```bash
python test_rag.py
```

---

## 📊 Evaluation & Test Metrics

Our evaluation suite (`test_rag.py`) tests 15 questions across three distinct categories:

1. **Category A (Direct Questions):** Verified exact lookup of attendance percentages, fees, and CGPA cutoffs.
2. **Category B (Multi-Chunk Questions):** Verified retrieval across multiple section clauses (e.g., placement tier rules combined with backlog policies).
3. **Category C (Out-of-Domain Refusal):** Verified that queries regarding non-existent topics (e.g., Quantum Physics syllabus, sports facilities) trigger the guardrail refusal:  
   *"I don't have enough information to answer that."*

---

## 🎓 Viva Questions & Key Answers

1. **Why use RAG instead of fine-tuning an LLM?**
   * *Answer:* Fine-tuning changes model weights, is computationally expensive, and requires full retraining whenever documents update. RAG injects current context dynamically into the prompt at runtime without altering model weights.

2. **Why use `sentence-transformers/all-MiniLM-L6-v2`?**
   * *Answer:* It maps sentences to a 384-dimensional dense vector space, balances high semantic retrieval accuracy with low computational overhead, and runs efficiently locally on CPU (~80 MB footprint).

3. **What is chunk overlap and why is it necessary?**
   * *Answer:* Chunk overlap (e.g., 64 tokens) preserves contextual continuity at boundaries so that key facts spanning across adjacent chunks are not split or lost during embedding.

---

## 📜 License & Acknowledgments
Developed for BTech Final Year Computer Science & Engineering Project Submission. Built with open-source tools: LlamaIndex, ChromaDB, Hugging Face, Groq, and Streamlit.
