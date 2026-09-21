import os
import sys

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List, Dict, Any, Optional
from src.ingestion import load_documents

from src.chunking import split_documents_into_chunks
from src.embeddings import get_embedding_model
from src.vector_store import create_vector_index, load_existing_index
from src.retriever import retrieve_top_k_chunks
from src.llm import get_llm
from src.prompts import get_rag_prompt_template

class SmartCampusRAGEngine:
    """
    Complete RAG Engine Pipeline orchestrating Ingestion, Embedding, Vector Storage,
    Retrieval, Prompting, and Open-Source LLM Response Generation.
    """

    def __init__(
        self,
        data_dir: str = "./data",
        db_path: str = "./chroma_db",
        force_rebuild: bool = False
    ):
        self.data_dir = data_dir
        self.db_path = db_path

        # 1. Initialize Embedding Model & LLM
        self.embed_model = get_embedding_model()
        self.llm = get_llm()
        self.prompt_template = get_rag_prompt_template()

        # 2. Build or Load Index
        self.index = None
        if not force_rebuild:
            self.index = load_existing_index(self.embed_model, self.db_path)

        if self.index is None:
            print("🔄 Building new ChromaDB vector index from data documents...")
            docs = load_documents(self.data_dir)
            if not docs:
                raise ValueError(f"No documents found in '{self.data_dir}'. Add PDF/TXT files first.")
            chunks = split_documents_into_chunks(docs)
            self.index = create_vector_index(chunks, self.embed_model, self.db_path)

        print("🚀 SmartCampus RAG Engine is ready!")

    def query(self, user_question: str, top_k: int = 3) -> Dict[str, Any]:
        """
        Executes end-to-end RAG query pipeline for a user question.

        Pipeline Steps:
        1. Retrieve top-k relevant chunks from ChromaDB.
        2. Format context string with retrieved content.
        3. Inject context into strict anti-hallucination prompt template.
        4. Synthesize answer using open-source LLM.
        5. Extract source document citations.

        Args:
            user_question (str): User query string.
            top_k (int): Number of matching chunks to retrieve.

        Returns:
            Dict containing:
                - question (str)
                - answer (str)
                - sources (List[str])
                - retrieved_chunks (List[Dict])
                - is_refusal (bool)
        """
        # Step 1: Retrieve top-k chunks
        retrieved_chunks = retrieve_top_k_chunks(self.index, user_question, top_k=top_k)

        if not retrieved_chunks:
            return {
                "question": user_question,
                "answer": "I don't have enough information to answer that.",
                "sources": [],
                "retrieved_chunks": [],
                "is_refusal": True
            }

        # Step 2: Format context string
        context_blocks = []
        source_files = set()
        for idx, chunk in enumerate(retrieved_chunks, start=1):
            source_files.add(chunk["file_name"])
            context_blocks.append(f"[Chunk {idx} | Source: {chunk['file_name']}]\n{chunk['text']}")

        context_str = "\n\n".join(context_blocks)

        # Step 3: Format Prompt
        formatted_prompt = self.prompt_template.format(
            context_str=context_str,
            query_str=user_question
        )

        # Step 4: Call LLM
        response = self.llm.complete(formatted_prompt)
        raw_answer = response.text.strip()

        # Step 5: Check refusal guardrail
        refusal_phrase = "I don't have enough information to answer that."
        is_refusal = refusal_phrase.lower() in raw_answer.lower()

        return {
            "question": user_question,
            "answer": raw_answer,
            "sources": [] if is_refusal else sorted(list(source_files)),
            "retrieved_chunks": [] if is_refusal else retrieved_chunks,
            "is_refusal": is_refusal
        }


if __name__ == "__main__":
    engine = SmartCampusRAGEngine()
    res = engine.query("What is the attendance condonation fee?")
    print(f"\nQ: {res['question']}")
    print(f"A:\n{res['answer']}")
    print(f"Sources: {res['sources']}")
