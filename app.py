import os
import sys
import warnings

# Suppress non-critical third-party warnings & force fast offline model loading
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"


import streamlit as st
from dotenv import load_dotenv


# Page Configuration
st.set_page_config(
    page_title="SmartCampus — RAG Knowledge Assistant",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .main-header h1 {
        color: #ffffff;
        font-weight: 700;
        margin-bottom: 0.3rem;
    }
    .main-header p {
        color: #e0e6ed;
        font-size: 1.05rem;
    }
    .badge-status {
        background-color: #28a745;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    .source-box {
        background-color: #f8f9fa;
        border-left: 4px solid #2a5298;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 0.5rem;
    }
    .refusal-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 0.75rem;
        border-radius: 4px;
        color: #856404;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

load_dotenv()

# Lazy Import of RAG Engine to avoid slow initial UI render
from src.rag_engine import SmartCampusRAGEngine

@st.cache_resource(show_spinner="Initializing RAG Vector Index & Embedding Models...")
def initialize_rag_engine():
    """Initializes and caches the RAG Engine instance."""
    return SmartCampusRAGEngine(data_dir="./data", db_path="./chroma_db", force_rebuild=False)

def main():
    # Sidebar
    with st.sidebar:
        st.title("🎓 Knowledge Base")
        st.caption("SmartCampus Academic Regulations")
        st.markdown("---")

        # Document Inventory Overview
        st.subheader("📁 Ingested Documents")
        data_dir = "./data"
        if os.path.exists(data_dir):
            files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.pdf', '.txt'))]
            if files:
                for f in files:
                    st.markdown(f"• `📄 {f}`")
            else:
                st.warning("No documents found in ./data")
        else:
            st.error("Data directory missing.")

        st.markdown("---")

        # System Control Buttons
        st.subheader("⚙️ System Actions")

        if st.button("🔄 Rebuild Vector Index", use_container_width=True):
            with st.spinner("Re-indexing ChromaDB..."):
                try:
                    rag_engine = SmartCampusRAGEngine(data_dir="./data", db_path="./chroma_db", force_rebuild=True)
                    st.cache_resource.clear()
                    st.success("Vector Store Index Rebuilt!")
                    st.rerun()
                except Exception as e:
                    st.error(f"Rebuild failed: {e}")

        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

        st.markdown("---")
        groq_key = os.getenv("GROQ_API_KEY", "")
        if groq_key and not groq_key.startswith("gsk_your_groq_api_key"):
            st.markdown("🟢 **LLM Status:** Groq Cloud API (Llama 3.1)")
        else:
            st.markdown("🟡 **LLM Status:** Fallback Engine (No API key)")
            st.caption("Add `GROQ_API_KEY` in `.env` for Groq Llama 3.1 generation.")

    # Header Banner
    st.markdown("""
    <div class="main-header">
        <h1>🎓 SmartCampus RAG Knowledge Assistant</h1>
        <p>Groundable QA Assistant for University Academic Regulations, Exam Rules & Placement Policies</p>
        <span class="badge-status">RAG Architecture Active (ChromaDB + LlamaIndex + Llama 3.1)</span>
    </div>
    """, unsafe_allow_html=True)

    # Initialize RAG Engine
    rag_engine = initialize_rag_engine()



    # Session State Chat Initialization
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {
                "role": "assistant",
                "content": "Hello! I am your SmartCampus Knowledge Assistant. Ask me anything about attendance rules, exam re-evaluation, grading, or placement eligibility policies.",
                "sources": [],
                "chunks": [],
                "is_refusal": False
            }
        ]

    # Quick Sample Question Chips
    st.markdown("#### 💡 Sample Questions to Try:")
    col1, col2, col3 = st.columns(3)
    preset_query = None
    if col1.button("📌 Minimum Attendance?"):
        preset_query = "What is the minimum attendance required for exam eligibility?"
    if col2.button("📌 Re-evaluation Fee?"):
        preset_query = "What is the fee and timeline for re-evaluation of exam papers?"
    if col3.button("📌 Placement CGPA?"):
        preset_query = "What is the minimum CGPA and backlog criteria for campus placements?"


    # Display Chat Messages
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            if msg.get("is_refusal"):
                st.markdown(f'<div class="refusal-box">⚠️ <b>Guardrail Triggered:</b> {msg["content"]}</div>', unsafe_allow_html=True)
            else:
                st.markdown(msg["content"])
                if msg.get("sources"):
                    st.caption(f"📚 **Sources Cited:** {', '.join(msg['sources'])}")
                if msg.get("chunks"):
                    with st.expander("🔍 View Retrieved Context Chunks & Similarity Scores"):
                        for idx, c in enumerate(msg["chunks"], start=1):
                            st.markdown(f"**Chunk #{idx}** | *Source:* `{c['file_name']}` | *Score:* `{c['score']}`")
                            st.code(c["text"], language="text")

    # Handle User Input
    user_input = st.chat_input("Ask a question about university policies...")
    query_to_run = preset_query or user_input

    if query_to_run:
        # Display User Message
        st.session_state.messages.append({"role": "user", "content": query_to_run})
        with st.chat_message("user"):
            st.markdown(query_to_run)

        # Generate Assistant Response
        with st.chat_message("assistant"):
            with st.spinner("Searching ChromaDB & Generating Answer..."):
                response_data = rag_engine.query(query_to_run, top_k=3)
                answer = response_data["answer"]
                sources = response_data["sources"]
                chunks = response_data["retrieved_chunks"]
                is_refusal = response_data["is_refusal"]

                # Render Answer
                if is_refusal:
                    st.markdown(f'<div class="refusal-box">⚠️ <b>Guardrail Triggered:</b> {answer}</div>', unsafe_allow_html=True)
                else:
                    st.markdown(answer)

                    # Render Sources and Context only for grounded answers
                    if sources:
                        st.markdown(f"📚 **Sources Cited:** {', '.join(sources)}")

                    if chunks:
                        with st.expander("🔍 View Retrieved Context Chunks & Similarity Scores"):
                            for idx, c in enumerate(chunks, start=1):
                                st.markdown(f"**Chunk #{idx}** | *Source:* `{c['file_name']}` | *Score:* `{c['score']}`")
                                st.code(c["text"], language="text")


        # Save to Session State
        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "chunks": chunks,
            "is_refusal": is_refusal
        })
        if preset_query:
            st.rerun()

if __name__ == "__main__":
    main()
