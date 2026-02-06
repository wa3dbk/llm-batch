"""Tests for llm_batch.data_loader."""

import json
import pytest

from llm_batch.data_loader import DataLoader, DataItem, load_data


@pytest.fixture
def tsv_file(tmp_path):
    f = tmp_path / "data.tsv"
    f.write_text("source\ttarget\nhello\tbonjour\nworld\tmonde\n")
    return str(f)


@pytest.fixture
def csv_file(tmp_path):
    f = tmp_path / "data.csv"
    f.write_text("source,target\nhello,bonjour\nworld,monde\n")
    return str(f)


@pytest.fixture
def jsonl_file(tmp_path):
    f = tmp_path / "data.jsonl"
    lines = [
        json.dumps({"source": "hello", "id": 1}),
        json.dumps({"source": "world", "id": 2}),
        json.dumps({"source": "foo", "id": 3}),
    ]
    f.write_text("\n".join(lines) + "\n")
    return str(f)


@pytest.fixture
def txt_file(tmp_path):
    f = tmp_path / "data.txt"
    f.write_text("line one\nline two\nline three\n")
    return str(f)


class TestDataLoaderTSV:
    def test_load(self, tsv_file):
        loader = DataLoader(tsv_file)
        data = loader.load()
        assert len(data) == 2
        assert data[0]["source"] == "hello"
        assert data[0]["target"] == "bonjour"

    def test_columns_auto_detect(self, tsv_file):
        loader = DataLoader(tsv_file)
        loader.load()
        assert loader.get_columns() == ["source", "target"]


class TestDataLoaderCSV:
    def test_load(self, csv_file):
        loader = DataLoader(csv_file)
        data = loader.load()
        assert len(data) == 2
        assert data[1]["source"] == "world"


class TestDataLoaderJSONL:
    def test_load(self, jsonl_file):
        loader = DataLoader(jsonl_file)
        data = loader.load()
        assert len(data) == 3
        assert data[0]["source"] == "hello"
        assert data[2]["id"] == 3

    def test_skip(self, jsonl_file):
        loader = DataLoader(jsonl_file, skip=1)
        data = loader.load()
        assert len(data) == 2
        assert data[0]["source"] == "world"

    def test_limit(self, jsonl_file):
        loader = DataLoader(jsonl_file, limit=2)
        data = loader.load()
        assert len(data) == 2

    def test_skip_and_limit(self, jsonl_file):
        loader = DataLoader(jsonl_file, skip=1, limit=1)
        data = loader.load()
        assert len(data) == 1
        assert data[0]["source"] == "world"


class TestDataLoaderTXT:
    def test_load(self, txt_file):
        loader = DataLoader(txt_file)
        data = loader.load()
        assert len(data) == 3
        assert data[0]["text"] == "line one"

    def test_custom_column(self, txt_file):
        loader = DataLoader(txt_file, columns=["sentence"])
        data = loader.load()
        assert data[0]["sentence"] == "line one"


class TestDataItem:
    def test_getitem(self):
        item = DataItem(index=0, data={"a": 1, "b": 2})
        assert item["a"] == 1
        assert item["missing"] is None

    def test_get_with_default(self):
        item = DataItem(index=0, data={"a": 1})
        assert item.get("a") == 1
        assert item.get("b", 99) == 99


class TestDataLoaderIteration:
    def test_iter(self, tsv_file):
        loader = DataLoader(tsv_file)
        items = list(loader)
        assert len(items) == 2

    def test_len(self, tsv_file):
        loader = DataLoader(tsv_file)
        assert len(loader) == 2

    def test_getitem(self, tsv_file):
        loader = DataLoader(tsv_file)
        item = loader[0]
        assert item["source"] == "hello"


class TestConvenienceFunction:
    def test_load_data(self, tsv_file):
        loader = load_data(tsv_file)
        assert len(loader) == 2


class TestPreview:
    def test_preview(self, tsv_file):
        loader = DataLoader(tsv_file)
        loader.load()
        preview = loader.preview(n=1)
        assert "2 items total" in preview
        assert "source" in preview
