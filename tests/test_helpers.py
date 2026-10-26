import importlib.util
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_FILE = PROJECT_ROOT / "app" / "main.py"


def test_main_file_exists():
    assert MAIN_FILE.is_file()


def test_main_file_has_required_features():
    source = MAIN_FILE.read_text(encoding="utf-8")
    required = [
        "def extract_audio(",
        "def transcribe(",
        "def build_search_index(",
        "def search_lecture(",
        "def extract_whiteboards(",
        "def make_concept_graph(",
        "def generate_quiz(",
        "def weak_concepts(",
        "def run_code(",
    ]
    for feature in required:
        assert feature in source, f"Missing feature: {feature}"


def test_python_syntax():
    source = MAIN_FILE.read_text(encoding="utf-8")
    compile(source, str(MAIN_FILE), "exec")
