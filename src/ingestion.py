import os
import sys

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from typing import List
from llama_index.core import SimpleDirectoryReader, Document


def load_documents(data_dir: str = "./data") -> List[Document]:
    """
    Reads all PDF and TXT documents from the specified directory.
    Preserves file metadata (filename, path) for source citation.

    Args:
        data_dir (str): Path to directory containing knowledge base documents.

    Returns:
        List[Document]: List of LlamaIndex Document objects.
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' does not exist.")

    files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.pdf', '.txt'))]
    if not files:
        print(f"⚠️ Warning: No PDF or TXT files found in '{data_dir}'.")
        return []

    print(f"📖 Loading {len(files)} document(s) from '{data_dir}'...")

    # SimpleDirectoryReader parses PDFs and TXT files automatically
    reader = SimpleDirectoryReader(
        input_dir=data_dir,
        required_exts=[".pdf", ".txt"],
        filename_as_id=True
    )
    documents = reader.load_data()

    print(f"✅ Successfully loaded {len(documents)} document page(s)/section(s).")
    for idx, doc in enumerate(documents, start=1):
        file_name = doc.metadata.get("file_name", "Unknown File")
        text_preview = doc.text[:100].replace("\n", " ")
        print(f"  [{idx}] File: {file_name} | Preview: '{text_preview}...'")

    return documents

if __name__ == "__main__":
    # Self-test block
    docs = load_documents("./data")
    print(f"\nSelf-test Passed: {len(docs)} documents ingested.")
