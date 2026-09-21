import os
import sys

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from typing import List
from llama_index.core import Document

from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode

def get_sentence_splitter(chunk_size: int = 512, chunk_overlap: int = 64) -> SentenceSplitter:
    """
    Creates a sentence-aware text splitter.

    Sentence-aware chunking breaks documents at sentence boundaries rather than
    arbitrary character cuts, preserving semantic context within each chunk.

    Args:
        chunk_size (int): Max number of tokens per chunk (default: 512).
        chunk_overlap (int): Token overlap between consecutive chunks (default: 64).

    Returns:
        SentenceSplitter: Configured LlamaIndex SentenceSplitter instance.
    """
    return SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

def split_documents_into_chunks(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 64
) -> List[BaseNode]:
    """
    Splits a list of Documents into smaller, semantic Nodes (chunks).

    Args:
        documents (List[Document]): Ingested raw documents.
        chunk_size (int): Max tokens per chunk.
        chunk_overlap (int): Token overlap.

    Returns:
        List[BaseNode]: List of chunk Nodes containing text and metadata.
    """
    splitter = get_sentence_splitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    nodes = splitter.get_nodes_from_documents(documents)
    print(f"🧩 Split {len(documents)} document(s) into {len(nodes)} semantic chunk(s).")
    return nodes

if __name__ == "__main__":
    from src.ingestion import load_documents
    docs = load_documents("./data")
    if docs:
        chunks = split_documents_into_chunks(docs)
        print(f"Sample Chunk [0] (File: {chunks[0].metadata.get('file_name')}):\n{chunks[0].get_content()[:200]}...")
