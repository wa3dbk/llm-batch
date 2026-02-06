"""
Shared utilities for LLM Inference CLI.
"""

from pathlib import Path


FORMAT_MAP = {
    ".tsv": "tsv",
    ".csv": "csv",
    ".jsonl": "jsonl",
    ".json": "jsonl",
    ".txt": "txt",
}


def detect_format(filepath: str, default: str = "tsv") -> str:
    """Detect file format from extension.

    Args:
        filepath: Path to the file.
        default: Format to return when the extension is unrecognized.

    Returns:
        One of "tsv", "csv", "jsonl", or "txt".
    """
    ext = Path(filepath).suffix.lower()
    return FORMAT_MAP.get(ext, default)
