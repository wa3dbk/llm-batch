"""Tests for llm_batch.template."""

import pytest

from llm_batch.template import PromptTemplate, PromptBuilder, get_template, TEMPLATES


class TestPromptTemplate:
    def test_basic_render(self):
        tpl = PromptTemplate("Translate: {source}")
        assert tpl.render({"source": "hello"}) == "Translate: hello"

    def test_placeholders(self):
        tpl = PromptTemplate("From {src} to {tgt}: {text}")
        assert sorted(tpl.placeholders) == ["src", "text", "tgt"]

    def test_render_missing_non_strict(self):
        tpl = PromptTemplate("Hello {name}, welcome to {place}")
        result = tpl.render({"name": "Alice"})
        assert "Alice" in result
        # Missing placeholder replaced with empty string
        assert "{place}" not in result

    def test_render_missing_strict(self):
        tpl = PromptTemplate("Hello {name}")
        with pytest.raises(ValueError, match="Missing values"):
            tpl.render({}, strict=True)

    def test_default_values(self):
        tpl = PromptTemplate("Language: {lang}", default_values={"lang": "English"})
        assert tpl.render({}) == "Language: English"

    def test_default_overridden_by_data(self):
        tpl = PromptTemplate("Language: {lang}", default_values={"lang": "English"})
        assert tpl.render({"lang": "French"}) == "Language: French"

    def test_system_prompt(self):
        tpl = PromptTemplate("Translate: {text}", system_prompt="You are a translator.")
        assert tpl.render_system() == "You are a translator."

    def test_system_prompt_with_placeholders(self):
        tpl = PromptTemplate("Q: {q}", system_prompt="You speak {lang}.")
        assert tpl.render_system({"lang": "French"}) == "You speak French."

    def test_chat_messages_no_tokenizer(self):
        tpl = PromptTemplate("Hello {name}", system_prompt="Be helpful.")
        msgs = tpl.render_chat_messages({"name": "Alice"})
        assert isinstance(msgs, list)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"
        assert msgs[1]["content"] == "Hello Alice"

    def test_load_from_file(self, tmp_path):
        f = tmp_path / "tpl.txt"
        f.write_text("Summarize: {text}")
        tpl = PromptTemplate(str(f))
        assert tpl.render({"text": "article"}) == "Summarize: article"

    def test_load_system_from_file(self, tmp_path):
        f = tmp_path / "sys.txt"
        f.write_text("You are a bot.")
        tpl = PromptTemplate("Hi", system_prompt=str(f))
        assert tpl.render_system() == "You are a bot."


class TestPromptBuilder:
    def test_basic_build(self):
        builder = PromptBuilder()
        builder.add_instruction("Translate this.")
        builder.add_input("{source}")
        tpl = builder.build()
        result = tpl.render({"source": "hello"})
        assert "Translate this." in result
        assert "hello" in result

    def test_with_system(self):
        builder = PromptBuilder()
        builder.add_system("Be concise.")
        builder.add_text("Q: {q}")
        tpl = builder.build()
        assert tpl.system_prompt == "Be concise."

    def test_output_prefix(self):
        builder = PromptBuilder()
        builder.add_text("Text: {text}")
        builder.add_output_prefix("Summary:")
        tpl = builder.build()
        result = tpl.render({"text": "article"})
        assert result.endswith("Summary:")


class TestBuiltinTemplates:
    def test_all_templates_exist(self):
        for name in ["translation", "translation_simple", "summarization", "qa", "classification", "completion"]:
            tpl = get_template(name)
            assert isinstance(tpl, PromptTemplate)

    def test_unknown_template_raises(self):
        with pytest.raises(ValueError, match="Unknown template"):
            get_template("nonexistent")
