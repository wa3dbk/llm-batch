"""Tests for llm_batch.output."""

import json
import pytest

from llm_batch.output import OutputProcessor, OutputWriter, InferenceResult, ResultCollector


class TestOutputProcessor:
    def test_strip(self):
        proc = OutputProcessor(strip=True)
        assert proc.process("  hello  ") == "hello"

    def test_no_strip(self):
        proc = OutputProcessor(strip=False)
        assert proc.process("  hello  ") == "  hello  "

    def test_stop_strings(self):
        proc = OutputProcessor(stop_strings=["<END>"])
        assert proc.process("hello world<END>extra") == "hello world"

    def test_extract_pattern_with_group(self):
        proc = OutputProcessor(extract_pattern=r"Answer:\s*(.*)")
        assert proc.process("Answer: 42") == "42"

    def test_extract_pattern_no_group(self):
        proc = OutputProcessor(extract_pattern=r"\d+")
        assert proc.process("The answer is 42.") == "42"

    def test_extract_pattern_no_match(self):
        proc = OutputProcessor(extract_pattern=r"MISSING:\s*(.*)")
        result = proc.process("no match here")
        assert result == "no match here"

    def test_combined_processing(self):
        proc = OutputProcessor(strip=True, stop_strings=["<|end|>"], extract_pattern=r"Result:\s*(.*)")
        text = "  Result: success<|end|>garbage  "
        assert proc.process(text) == "success"


class TestOutputWriterJSONL:
    def test_write_jsonl(self, tmp_path):
        outpath = tmp_path / "out.jsonl"
        writer = OutputWriter(str(outpath), format="jsonl")
        writer.open()
        writer.write(InferenceResult(
            index=0,
            input_data={"source": "hi"},
            prompt="Translate: hi",
            output="salut",
            raw_output="salut",
        ))
        writer.close()

        lines = outpath.read_text().strip().splitlines()
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["output"] == "salut"
        assert data["index"] == 0

    def test_include_input(self, tmp_path):
        outpath = tmp_path / "out.jsonl"
        writer = OutputWriter(str(outpath), format="jsonl", include_input=True)
        writer.open()
        writer.write(InferenceResult(
            index=0,
            input_data={"source": "hi"},
            prompt="p",
            output="salut",
            raw_output="salut",
        ))
        writer.close()

        data = json.loads(outpath.read_text().strip())
        assert data["input"] == {"source": "hi"}

    def test_include_prompt(self, tmp_path):
        outpath = tmp_path / "out.jsonl"
        writer = OutputWriter(str(outpath), format="jsonl", include_prompt=True)
        writer.open()
        writer.write(InferenceResult(
            index=0,
            input_data={},
            prompt="my prompt",
            output="out",
            raw_output="out",
        ))
        writer.close()

        data = json.loads(outpath.read_text().strip())
        assert data["prompt"] == "my prompt"


class TestOutputWriterTSV:
    def test_write_tsv(self, tmp_path):
        outpath = tmp_path / "out.tsv"
        writer = OutputWriter(str(outpath), format="tsv")
        writer.open()
        writer.write(InferenceResult(
            index=0,
            input_data={"source": "hi"},
            prompt="p",
            output="salut",
            raw_output="salut",
        ))
        writer.close()

        lines = outpath.read_text().strip().splitlines()
        assert len(lines) == 2  # header + data
        assert "output" in lines[0]
        assert "salut" in lines[1]


class TestOutputWriterAppend:
    def test_append_mode(self, tmp_path):
        outpath = tmp_path / "out.jsonl"
        # Write first item
        writer = OutputWriter(str(outpath), format="jsonl")
        writer.open()
        writer.write(InferenceResult(0, {}, "p", "first", "first"))
        writer.close()

        # Append second item
        writer2 = OutputWriter(str(outpath), format="jsonl")
        writer2.open(append=True)
        writer2.write(InferenceResult(1, {}, "p", "second", "second"))
        writer2.close()

        lines = outpath.read_text().strip().splitlines()
        assert len(lines) == 2
        assert json.loads(lines[0])["output"] == "first"
        assert json.loads(lines[1])["output"] == "second"


class TestCheckpoint:
    def test_checkpoint_saved(self, tmp_path):
        outpath = tmp_path / "out.jsonl"
        writer = OutputWriter(str(outpath), format="jsonl", checkpoint_every=1)
        writer.open()
        writer.write(InferenceResult(0, {"s": "a"}, "p", "out", "out"))
        writer.close()

        cp_path = tmp_path / "out.jsonl.checkpoint"
        assert cp_path.exists()
        cp = json.loads(cp_path.read_text())
        assert cp["count"] == 1
        assert cp["last_index"] == 0

    def test_checkpoint_includes_config(self, tmp_path):
        outpath = tmp_path / "out.jsonl"
        cfg = {"model": "test", "batch_size": 4}
        writer = OutputWriter(str(outpath), format="jsonl", checkpoint_every=1, config_dict=cfg)
        writer.open()
        writer.write(InferenceResult(0, {}, "p", "out", "out"))
        writer.close()

        cp = json.loads((tmp_path / "out.jsonl.checkpoint").read_text())
        assert cp["config"]["model"] == "test"

    def test_load_checkpoint(self, tmp_path):
        cp_path = tmp_path / "test.checkpoint"
        cp_path.write_text(json.dumps({"count": 5, "last_index": 4, "filepath": "out.jsonl"}))
        data = OutputWriter.load_checkpoint(str(cp_path))
        assert data["count"] == 5

    def test_load_checkpoint_not_found(self):
        with pytest.raises(FileNotFoundError):
            OutputWriter.load_checkpoint("/nonexistent/path.checkpoint")


class TestResultCollector:
    def test_collect_outputs(self):
        rc = ResultCollector()
        rc.add(InferenceResult(0, {}, "p", "a", "a"))
        rc.add(InferenceResult(1, {}, "p", "b", "b"))
        assert rc.get_outputs() == ["a", "b"]
        assert len(rc) == 2

    def test_get_inputs(self):
        rc = ResultCollector()
        rc.add(InferenceResult(0, {"src": "x"}, "p", "a", "a"))
        rc.add(InferenceResult(1, {"src": "y"}, "p", "b", "b"))
        assert rc.get_inputs("src") == ["x", "y"]
