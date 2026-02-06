"""Tests for llm_batch.config."""

import json
import yaml
import pytest

from llm_batch.config import InferenceConfig


def test_default_values():
    cfg = InferenceConfig()
    assert cfg.model == ""
    assert cfg.quantization == "4bit"
    assert cfg.backend == "auto"
    assert cfg.batch_size == 1
    assert cfg.temperature == 0.7


def test_auto_detect_input_format():
    cfg = InferenceConfig(input_file="data.csv")
    assert cfg.input_format == "csv"


def test_auto_detect_output_format():
    cfg = InferenceConfig(output_file="results.jsonl")
    assert cfg.output_format == "jsonl"


def test_stop_strings_parsed_from_str():
    cfg = InferenceConfig(stop_strings="foo, bar, baz")
    assert cfg.stop_strings == ["foo", "bar", "baz"]


def test_stop_strings_list_unchanged():
    cfg = InferenceConfig(stop_strings=["a", "b"])
    assert cfg.stop_strings == ["a", "b"]


def test_template_not_loaded_from_file_in_config(tmp_path):
    """Config should NOT load template file contents — PromptTemplate does that."""
    tpl_file = tmp_path / "prompt.md"
    tpl_file.write_text("Translate: {source}")
    cfg = InferenceConfig(template=str(tpl_file))
    # Should still be the file path, not the file contents
    assert cfg.template == str(tpl_file)


def test_to_dict():
    cfg = InferenceConfig(model="test-model", temperature=0.5)
    d = cfg.to_dict()
    assert d["model"] == "test-model"
    assert d["temperature"] == 0.5


def test_from_file_json(tmp_path):
    cfg_file = tmp_path / "config.json"
    cfg_file.write_text(json.dumps({"model": "my-model", "batch_size": 4}))
    cfg = InferenceConfig.from_file(str(cfg_file))
    assert cfg.model == "my-model"
    assert cfg.batch_size == 4


def test_from_file_yaml(tmp_path):
    cfg_file = tmp_path / "config.yaml"
    cfg_file.write_text(yaml.dump({"model": "my-model", "temperature": 0.2}))
    cfg = InferenceConfig.from_file(str(cfg_file))
    assert cfg.model == "my-model"
    assert cfg.temperature == 0.2


def test_save_and_load_roundtrip(tmp_path):
    cfg = InferenceConfig(model="roundtrip-model", max_new_tokens=512)
    path = str(tmp_path / "cfg.json")
    cfg.save(path)
    loaded = InferenceConfig.from_file(path)
    assert loaded.model == "roundtrip-model"
    assert loaded.max_new_tokens == 512
