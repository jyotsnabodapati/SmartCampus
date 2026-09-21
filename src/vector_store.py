import os
import sys

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import chromadb
from typing import List, Optional

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.core.schema import BaseNode
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding

def get_chroma_vector_store(
    db_path: str = "./chroma_db",
    collection_name: str = "smart_campus_kb",
    clear_existing: bool = False
) -> ChromaVectorStore:
    """
    Initializes a persistent ChromaDB database client and collection.

    Args:
        db_path (str): Directory where ChromaDB data will be saved.
        collection_name (str): Name of the ChromaDB collection.
        clear_existing (bool): If True, deletes existing collection to avoid duplicate vectors.

    Returns:
        ChromaVectorStore: LlamaIndex Chroma vector store instance.
    """
    os.makedirs(db_path, exist_ok=True)
    print(f"🗄️ Connecting to Persistent ChromaDB at '{db_path}'...")
    db_client = chromadb.PersistentClient(path=db_path)

    if clear_existing:
        try:
            db_client.delete_collection(collection_name)
            print(f"🧹 Cleared existing collection '{collection_name}'.")
        except Exception:
            pass

    chroma_collection = db_client.get_or_create_collection(collection_name)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    print(f"✅ Connected to collection '{collection_name}' (Stored items: {chroma_collection.count()}).")
    return vector_store


def create_vector_index(
    nodes: List[BaseNode],
    embed_model: HuggingFaceEmbedding,
    db_path: str = "./chroma_db",
    collection_name: str = "smart_campus_kb",
    clear_existing: bool = True
) -> VectorStoreIndex:
    """
    Stores document chunks (nodes) and their vector embeddings into ChromaDB
    and returns a LlamaIndex VectorStoreIndex.

    Args:
        nodes (List[BaseNode]): List of chunk nodes.
        embed_model (HuggingFaceEmbedding): Embedding model instance.
        db_path (str): ChromaDB folder path.
        collection_name (str): Collection name.
        clear_existing (bool): Clears old vectors before indexing new ones.

    Returns:
        VectorStoreIndex: Indexed Vector Store ready for retrieval.
    """
    vector_store = get_chroma_vector_store(db_path, collection_name, clear_existing=clear_existing)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)


    print(f"⚡ Indexing {len(nodes)} chunks into ChromaDB...")
    index = VectorStoreIndex(
        nodes=nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True
    )
    print("✅ Indexing complete.")
    return index

def load_existing_index(
    embed_model: HuggingFaceEmbedding,
    db_path: str = "./chroma_db",
    collection_name: str = "smart_campus_kb"
) -> Optional[VectorStoreIndex]:
    """
    Loads an existing VectorStoreIndex from ChromaDB without re-indexing.

    Args:
        embed_model (HuggingFaceEmbedding): Embedding model instance.
        db_path (str): Path to persistent ChromaDB directory.
        collection_name (str): Name of the collection.

    Returns:
        Optional[VectorStoreIndex]: Loaded index or None if collection is empty.
    """
    os.makedirs(db_path, exist_ok=True)
    db_client = chromadb.PersistentClient(path=db_path)
    
    try:
        chroma_collection = db_client.get_collection(collection_name)
        if chroma_collection.count() == 0:
            return None
        vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
        index = VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            embed_model=embed_model
        )
        print(f"✅ Loaded existing ChromaDB index with {chroma_collection.count()} vectors.")
        return index
    except Exception:
        return None

if __name__ == "__main__":
    from src.ingestion import load_documents
    from src.chunking import split_documents_into_chunks
    from src.embeddings import get_embedding_model

    docs = load_documents("./data")
    if docs:
        chunks = split_documents_into_chunks(docs)
        embed_mdl = get_embedding_model()
        idx = create_vector_index(chunks, embed_mdl)
        print("Vector Store Test Passed.")
