import os
import sys
import tempfile
import warnings
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Suppress non-critical third-party warnings and enforce PyTorch backend
warnings.filterwarnings("ignore")
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
os.environ["TRANSFORMERS_NO_TF"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["USE_TF"] = "0"
os.environ["USE_TORCH"] = "1"
os.environ.setdefault("KERAS_HOME", os.path.join(tempfile.gettempdir(), ".keras"))

# Ensure NLTK stopwords are downloaded in isolated user environment
try:
    import nltk
    nltk.download("stopwords", quiet=True)
except Exception:
    pass

from llama_index.core import Document
from llama_index.core.embeddings import BaseEmbedding
from src.llm import MockLocalLLM


class FastMockEmbedding(BaseEmbedding):
    """
    Ultra-fast mock embedding model returning 384-dimensional vectors.
    Allows unit tests to run in milliseconds without downloading models.
    """
    model_name: str = "FastMockEmbedding"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    @classmethod
    def class_name(cls) -> str:
        return "FastMockEmbedding"

    def _get_query_embedding(self, query: str):
        return [0.05] * 384

    def _get_text_embedding(self, text: str):
        return [0.05] * 384

    async def _aget_query_embedding(self, query: str):
        return [0.05] * 384

    async def _aget_text_embedding(self, text: str):
        return [0.05] * 384


@pytest.fixture
def sample_documents():
    """Provides sample Document objects with university policy content."""
    return [
        Document(
            text="Minimum attendance required is 75 percent for exam eligibility. Shortage below 65 percent results in semester detention.",
            metadata={"file_name": "Academic_Regulations.txt"}
        ),
        Document(
            text="Answer script re-evaluation fee is Rs. 750 per subject. Application must be submitted within 15 days.",
            metadata={"file_name": "Examination_Rules.txt"}
        ),
        Document(
            text="Minimum CGPA required for campus placement is 6.5 with zero active backlogs. Tier 1 Dream Tier cutoff is 12 LPA and above.",
            metadata={"file_name": "Placement_Policy.txt"}
        )
    ]


@pytest.fixture
def fast_embed_model():
    """Returns an ultra-fast mock embedding model."""
    return FastMockEmbedding()


@pytest.fixture
def mock_llm():
    """Returns the SmartCampus MockLocalLLM instance."""
    return MockLocalLLM()


@pytest.fixture
def temp_data_dir(tmp_path):
    """Creates a temporary isolated data directory with sample policy files."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    sample_file = data_dir / "Academic_Regulations.txt"
    sample_file.write_text("Minimum attendance required is 75 percent. Condonation fee is Rs. 1,500.", encoding="utf-8")
    return str(data_dir)


@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provides an isolated temporary directory for ChromaDB tests, preserving production DB."""
    db_dir = tmp_path / "test_chroma_db"
    db_dir.mkdir()
    return str(db_dir)
