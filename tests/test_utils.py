"""Tests for llm_batch.utils."""

from llm_batch.utils import detect_format


def test_detect_tsv():
    assert detect_format("data.tsv") == "tsv"


def test_detect_csv():
    assert detect_format("data.csv") == "csv"


def test_detect_jsonl():
    assert detect_format("data.jsonl") == "jsonl"


def test_detect_json_maps_to_jsonl():
    assert detect_format("data.json") == "jsonl"


def test_detect_txt():
    assert detect_format("data.txt") == "txt"


def test_detect_unknown_returns_default():
    assert detect_format("data.xyz") == "tsv"


def test_detect_custom_default():
    assert detect_format("data.xyz", default="csv") == "csv"


def test_detect_case_insensitive():
    assert detect_format("DATA.TSV") == "tsv"
    assert detect_format("file.Jsonl") == "jsonl"
