import os
import sys
import warnings

import os
import sys
import glob
import tempfile
import warnings

# Suppress non-critical third-party warnings & configure safe paths
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("KERAS_HOME", os.path.join(tempfile.gettempdir(), ".keras"))

# Ensure workspace root is in sys.path for direct execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

from llama_index.embeddings.huggingface import HuggingFaceEmbedding


def resolve_model_path(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> str:
    """
    Resolves the model name to a local snapshot directory if available,
    allowing reliable offline loading without network lookups.
    """
    if os.path.isdir(model_name):
        return model_name

    hub_name = "models--" + model_name.replace("/", "--")
    cache_hub = os.path.expanduser(os.path.join("~", ".cache", "huggingface", "hub", hub_name, "snapshots"))
    if os.path.isdir(cache_hub):
        snapshots = [
            os.path.join(cache_hub, s)
            for s in os.listdir(cache_hub)
            if os.path.isdir(os.path.join(cache_hub, s))
        ]
        for snap in snapshots:
            if os.path.exists(os.path.join(snap, "model.safetensors")) or os.path.exists(
                os.path.join(snap, "pytorch_model.bin")
            ):
                return snap

    return model_name


def get_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2") -> HuggingFaceEmbedding:
    """
    Initializes and returns the Hugging Face SentenceTransformer embedding model.

    Model Details:
    - Model: sentence-transformers/all-MiniLM-L6-v2
    - Dimensionality: 384 dimensions
    - Memory footprint: ~80 MB
    - Speed: Extremely fast CPU execution

    Args:
        model_name (str): Hugging Face model identifier or local path.

    Returns:
        HuggingFaceEmbedding: Configured LlamaIndex embedding model wrapper.
    """
    resolved_path = resolve_model_path(model_name)
    print(f"🧠 Loading Embedding Model: '{model_name}'...")
    embed_model = HuggingFaceEmbedding(model_name=resolved_path)
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
