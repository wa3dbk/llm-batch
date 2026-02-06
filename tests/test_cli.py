"""Tests for llm_batch.cli."""

import pytest
from unittest.mock import patch

from llm_batch.cli import parse_args, main


class TestParseArgs:
    def test_basic_args(self):
        args = parse_args.__wrapped__ if hasattr(parse_args, '__wrapped__') else None
        # parse_args uses sys.argv, so we patch it
        with patch("sys.argv", ["llm-batch", "-m", "test-model", "-i", "in.tsv", "-t", "Translate: {s}", "-o", "out.tsv"]):
            args = parse_args()
        assert args.model == "test-model"
        assert args.input == "in.tsv"
        assert args.template == "Translate: {s}"
        assert args.output == "out.tsv"

    def test_defaults(self):
        with patch("sys.argv", ["llm-batch", "-m", "m", "-i", "i", "-t", "t", "-o", "o"]):
            args = parse_args()
        assert args.quantization == "4bit"
        assert args.backend == "auto"
        assert args.batch_size == 1
        assert args.temperature == 0.7
        assert args.max_tokens == 256

    def test_list_models_flag(self):
        with patch("sys.argv", ["llm-batch", "--list-models"]):
            args = parse_args()
        assert args.list_models is True

    def test_num_workers_removed(self):
        """num_workers was removed — make sure the flag is gone."""
        with patch("sys.argv", ["llm-batch", "--num-workers", "4"]):
            with pytest.raises(SystemExit):
                parse_args()


class TestMain:
    def test_list_models_returns_zero(self, capsys):
        with patch("sys.argv", ["llm-batch", "--list-models"]):
            ret = main()
        assert ret == 0
        captured = capsys.readouterr()
        assert "RECOMMENDED MODELS" in captured.out

    def test_missing_model_returns_error(self):
        with patch("sys.argv", ["llm-batch", "-i", "in.tsv", "-t", "t", "-o", "o"]):
            ret = main()
        assert ret == 1

    def test_missing_input_returns_error(self):
        with patch("sys.argv", ["llm-batch", "-m", "m", "-t", "t", "-o", "o"]):
            ret = main()
        assert ret == 1

    def test_missing_template_returns_error(self):
        with patch("sys.argv", ["llm-batch", "-m", "m", "-i", "i", "-o", "o"]):
            ret = main()
        assert ret == 1

    def test_missing_output_returns_error(self):
        with patch("sys.argv", ["llm-batch", "-m", "m", "-i", "i", "-t", "t"]):
            ret = main()
        assert ret == 1

    def test_dry_run(self, tmp_path, capsys):
        infile = tmp_path / "in.tsv"
        infile.write_text("source\nhello\n")
        with patch("sys.argv", [
            "llm-batch", "-m", "model", "-i", str(infile),
            "-t", "Translate: {source}", "-o", "out.tsv", "--dry-run"
        ]):
            ret = main()
        assert ret == 0
        captured = capsys.readouterr()
        assert "DRY RUN" in captured.out
