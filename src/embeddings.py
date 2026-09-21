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


# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from llama_index.embeddings.huggingface import HuggingFaceEmbedding



def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbedding:
    """
    Initializes and returns the Hugging Face SentenceTransformer embedding model.

    Model Details:
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - Dimensionality: 384 dimensions
    - Memory footprint: ~80 MB
    - Speed: Extremely fast CPU execution

    Args:
        model_name (str): Hugging Face model identifier.

    Returns:
        HuggingFaceEmbedding: Configured LlamaIndex embedding model wrapper.
    """
    print(f"🧠 Loading Embedding Model: '{model_name}'...")
    embed_model = HuggingFaceEmbedding(model_name=model_name)
    print("✅ Embedding model loaded successfully.")
    return embed_model

def test_embedding():
    """Generates a test vector and prints its dimension to verify correctness."""
    embed_model = get_embedding_model()
    sample_text = "Minimum attendance required is 75 percent for exam eligibility."
    vector = embed_model.get_text_embedding(sample_text)
    print(f"Vector generated for sample text.")
    print(f"Vector Dimension: {len(vector)} (Expected: 384)")
    print(f"First 5 vector values: {vector[:5]}")
    return embed_model

if __name__ == "__main__":
    test_embedding()
