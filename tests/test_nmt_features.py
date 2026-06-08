"""Tests for NMT-oriented features added based on Jon et al. (AbjadNLP 2026).

Covers:
- max_length_ratio output cropping
- Built-in NMT templates (nmt, nmt_general, nmt_ar2en)
- NMT system prompt
- CLI --max-length-ratio flag
- Built-in template name resolution in the engine
"""

import json
import pytest
from unittest.mock import patch

from llm_batch.output import OutputProcessor
from llm_batch.template import PromptTemplate, TEMPLATES, get_template, _NMT_SYSTEM_PROMPT
from llm_batch.config import InferenceConfig
from llm_batch.cli import parse_args


# ---------------------------------------------------------------------------
# OutputProcessor: max_length_ratio
# ---------------------------------------------------------------------------

class TestMaxLengthRatio:
    def test_no_ratio_no_cropping(self):
        proc = OutputProcessor(max_length_ratio=None)
        output = "a" * 1000
        assert proc.process(output, source_len=10) == output

    def test_ratio_crops_long_output(self):
        proc = OutputProcessor(max_length_ratio=5.0)
        output = "x" * 100
        result = proc.process(output, source_len=10)
        assert len(result) == 50  # 10 * 5.0

    def test_ratio_keeps_short_output(self):
        proc = OutputProcessor(max_length_ratio=5.0)
        output = "short"
        result = proc.process(output, source_len=10)
        assert result == "short"

    def test_ratio_with_zero_source_len(self):
        """Should not crash when source_len is 0."""
        proc = OutputProcessor(max_length_ratio=5.0)
        output = "some output"
        result = proc.process(output, source_len=0)
        assert result == "some output"

    def test_ratio_with_none_source_len(self):
        """Should not crash when source_len is None (backward compat)."""
        proc = OutputProcessor(max_length_ratio=5.0)
        output = "some output"
        result = proc.process(output, source_len=None)
        assert result == "some output"

    def test_ratio_combined_with_stop_strings(self):
        proc = OutputProcessor(max_length_ratio=5.0, stop_strings=["<END>"])
        output = "hello world<END>" + "x" * 100
        result = proc.process(output, source_len=10)
        assert result == "hello world"

    def test_ratio_combined_with_extract_pattern(self):
        proc = OutputProcessor(max_length_ratio=10.0, extract_pattern=r"Result:\s*(.*)")
        output = "Result: the answer" + " " * 200
        result = proc.process(output, source_len=5)
        # After cropping to 50 chars, pattern extraction should still work
        assert "the answer" in result

    def test_backward_compat_process_without_source_len(self):
        """Old code calling process(output) without source_len should still work."""
        proc = OutputProcessor(max_length_ratio=5.0)
        result = proc.process("hello")
        assert result == "hello"


# ---------------------------------------------------------------------------
# OutputProcessor: first_line_only
# ---------------------------------------------------------------------------

class TestFirstLineOnly:
    def test_keeps_single_line(self):
        proc = OutputProcessor(first_line_only=True)
        assert proc.process("This is the translation.") == "This is the translation."

    def test_strips_commentary(self):
        proc = OutputProcessor(first_line_only=True)
        output = 'This is the first sentence.\n"It seems like you meant something else.\n\nIf you could provide more context..."'
        assert proc.process(output) == "This is the first sentence."

    def test_skips_leading_blank_lines(self):
        proc = OutputProcessor(first_line_only=True)
        output = "\n\n  \nActual translation here\nExtra stuff"
        assert proc.process(output) == "Actual translation here"

    def test_disabled_by_default(self):
        proc = OutputProcessor()
        output = "line1\nline2"
        assert proc.process(output) == "line1\nline2"

    def test_combined_with_stop_strings(self):
        proc = OutputProcessor(first_line_only=True, stop_strings=["<END>"])
        output = "Translation here<END>garbage\nmore garbage"
        assert proc.process(output) == "Translation here"

    def test_combined_with_length_ratio(self):
        proc = OutputProcessor(first_line_only=True, max_length_ratio=5.0)
        output = "x" * 100 + "\ncommentary"
        result = proc.process(output, source_len=10)
        assert len(result) == 50

    def test_real_world_verbose_qwen(self):
        """Reproduce the actual Qwen verbose output seen in translations.tsv."""
        proc = OutputProcessor(first_line_only=True)
        output = (
            '"It seems like you might have made a small typo or mix-up in your '
            'request. If you meant "ترجم هذه باللغة" it translates to '
            '"Translate this into the language" in English.\n\n'
            "If you could provide more context or clarify what exactly you "
            'want translated, I\'d be happy to help!"'
        )
        result = proc.process(output)
        # Should keep only the first line, even if it's noisy
        assert "\n" not in result
        assert "If you could provide" not in result


# ---------------------------------------------------------------------------
# NMT Templates
# ---------------------------------------------------------------------------

class TestNMTTemplates:
    def test_nmt_template_exists(self):
        assert "nmt" in TEMPLATES

    def test_nmt_general_template_exists(self):
        assert "nmt_general" in TEMPLATES

    def test_nmt_ar2en_template_exists(self):
        assert "nmt_ar2en" in TEMPLATES

    def test_nmt_template_has_system_prompt(self):
        tpl = TEMPLATES["nmt"]
        assert tpl.system_prompt is not None
        assert "no explanations" in tpl.system_prompt

    def test_nmt_template_render_dialect(self):
        tpl = TEMPLATES["nmt"]
        result = tpl.render({
            "target_language": "Egyptian Arabic",
            "source": "How are you?",
        })
        assert "Egyptian Arabic" in result
        assert "How are you?" in result
        assert "only print out the translation" in result

    def test_nmt_general_render(self):
        tpl = TEMPLATES["nmt_general"]
        result = tpl.render({"source": "Hello world"})
        assert "Arabic" in result
        assert "Hello world" in result

    def test_nmt_ar2en_render(self):
        tpl = TEMPLATES["nmt_ar2en"]
        result = tpl.render({"source": "مرحبا"})
        assert "English" in result
        assert "مرحبا" in result

    def test_nmt_system_prompt_content(self):
        assert "professional" in _NMT_SYSTEM_PROMPT
        assert "no explanations" in _NMT_SYSTEM_PROMPT
        assert "formatting" in _NMT_SYSTEM_PROMPT

    def test_get_template_nmt(self):
        tpl = get_template("nmt")
        assert isinstance(tpl, PromptTemplate)

    def test_get_template_nmt_ar2en(self):
        tpl = get_template("nmt_ar2en")
        assert isinstance(tpl, PromptTemplate)

    def test_all_nmt_templates_have_system_prompt(self):
        for name in ["nmt", "nmt_general", "nmt_ar2en"]:
            tpl = TEMPLATES[name]
            assert tpl.system_prompt is not None


# ---------------------------------------------------------------------------
# NMT template files
# ---------------------------------------------------------------------------

class TestNMTTemplateFiles:
    def test_system_nmt_file(self):
        tpl = PromptTemplate(
            template="Translate: {source}",
            system_prompt="templates/system_nmt.txt",
        )
        assert "professional" in tpl.system_prompt
        assert "no explanations" in tpl.system_prompt

    def test_nmt_dialect_file(self):
        tpl = PromptTemplate(template="templates/nmt_dialect.md")
        result = tpl.render({
            "target_language": "Tunisian Arabic",
            "source": "Good morning",
        })
        assert "Tunisian Arabic" in result
        assert "Good morning" in result

    def test_nmt_ar2en_file(self):
        tpl = PromptTemplate(template="templates/nmt_ar2en.md")
        result = tpl.render({"source": "مرحبا"})
        assert "English" in result

    def test_nmt_en2ar_file(self):
        tpl = PromptTemplate(template="templates/nmt_en2ar.md")
        result = tpl.render({"source": "Hello"})
        assert "Arabic" in result


# ---------------------------------------------------------------------------
# Config: max_length_ratio
# ---------------------------------------------------------------------------

class TestConfigMaxLengthRatio:
    def test_default_is_none(self):
        cfg = InferenceConfig()
        assert cfg.max_length_ratio is None

    def test_set_ratio(self):
        cfg = InferenceConfig(max_length_ratio=5.0)
        assert cfg.max_length_ratio == 5.0

    def test_to_dict_includes_ratio(self):
        cfg = InferenceConfig(max_length_ratio=3.0)
        d = cfg.to_dict()
        assert d["max_length_ratio"] == 3.0

    def test_from_file_with_ratio(self, tmp_path):
        cfg_file = tmp_path / "cfg.json"
        cfg_file.write_text(json.dumps({"model": "test", "max_length_ratio": 5.0}))
        cfg = InferenceConfig.from_file(str(cfg_file))
        assert cfg.max_length_ratio == 5.0


# ---------------------------------------------------------------------------
# CLI: --max-length-ratio
# ---------------------------------------------------------------------------

class TestCLIMaxLengthRatio:
    def test_flag_parsed(self):
        with patch("sys.argv", [
            "llm-batch", "-m", "m", "-i", "i", "-t", "t", "-o", "o",
            "--max-length-ratio", "5.0",
        ]):
            args = parse_args()
        assert args.max_length_ratio == 5.0

    def test_flag_default_none(self):
        with patch("sys.argv", [
            "llm-batch", "-m", "m", "-i", "i", "-t", "t", "-o", "o",
        ]):
            args = parse_args()
        assert args.max_length_ratio is None

    def test_config_from_args_with_ratio(self):
        with patch("sys.argv", [
            "llm-batch", "-m", "m", "-i", "i.tsv", "-t", "t", "-o", "o.tsv",
            "--max-length-ratio", "3.5",
        ]):
            args = parse_args()
        cfg = InferenceConfig.from_args(args)
        assert cfg.max_length_ratio == 3.5


# ---------------------------------------------------------------------------
# CLI: --list-models includes NMT section
# ---------------------------------------------------------------------------

class TestListModelsNMT:
    def test_list_models_mentions_jais(self, capsys):
        from llm_batch.cli import list_models
        list_models()
        out = capsys.readouterr().out
        assert "Jais-2-70B-Chat" in out

    def test_list_models_mentions_aya(self, capsys):
        from llm_batch.cli import list_models
        list_models()
        out = capsys.readouterr().out
        assert "aya-expanse" in out

    def test_list_models_mentions_nmt_templates(self, capsys):
        from llm_batch.cli import list_models
        list_models()
        out = capsys.readouterr().out
        assert "nmt" in out
        assert "nmt_ar2en" in out
