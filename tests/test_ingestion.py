import os
import pytest
from src.ingestion import load_documents


def test_load_documents_from_project_data():
    """Verify that documents from the existing ./data directory load properly."""
    data_dir = "./data"
    assert os.path.exists(data_dir), "./data directory must exist"
    
    docs = load_documents(data_dir)
    assert len(docs) > 0, "Should load at least one document from ./data"
    
    # Verify metadata preservation
    file_names = {doc.metadata.get("file_name") for doc in docs}
    assert any("Academic_Regulations" in f for f in file_names if f)
    assert any("Examination_Rules" in f for f in file_names if f)
    assert any("Placement_Policy" in f for f in file_names if f)


def test_load_documents_missing_directory():
    """Verify that a non-existent directory raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        load_documents("./non_existent_data_directory_xyz")


def test_load_documents_empty_directory(tmp_path):
    """Verify that an empty directory returns an empty list without raising."""
    empty_dir = tmp_path / "empty_dir"
    empty_dir.mkdir()
    docs = load_documents(str(empty_dir))
    assert docs == []


def test_load_documents_preserves_text_and_metadata(temp_data_dir):
    """Verify that document text and file_name metadata are preserved."""
    docs = load_documents(temp_data_dir)
    assert len(docs) == 1
    assert "attendance required is 75 percent" in docs[0].text
    assert docs[0].metadata.get("file_name") == "Academic_Regulations.txt"

