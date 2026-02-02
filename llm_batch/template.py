"""
Prompt template handling for LLM Inference CLI.

Supports:
- String templates with {placeholder} syntax
- Template files (.txt, .md)
- Chat templates (system + user messages)
- Variable substitution from data items
"""

import re
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from string import Formatter


class PromptTemplate:
    """
    Handles prompt template parsing and rendering.
    
    Supports placeholders like {column_name} that are replaced
    with values from data items.
    
    Example:
        template = PromptTemplate("Translate to English: {source}")
        prompt = template.render({"source": "مرحبا"})
        # Result: "Translate to English: مرحبا"
    """
    
    def __init__(
        self,
        template: str,
        system_prompt: Optional[str] = None,
        default_values: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize prompt template.
        
        Args:
            template: Template string or path to template file
            system_prompt: Optional system prompt for chat models
            default_values: Default values for placeholders
        """
        # Load from file if path exists
        if Path(template).exists():
            template = Path(template).read_text(encoding="utf-8")
        
        if system_prompt and Path(system_prompt).exists():
            system_prompt = Path(system_prompt).read_text(encoding="utf-8")
        
        self.template = template.strip()
        self.system_prompt = system_prompt.strip() if system_prompt else None
        self.default_values = default_values or {}
        
        # Extract placeholder names
        self._placeholders = self._extract_placeholders(self.template)
        if self.system_prompt:
            self._placeholders.update(self._extract_placeholders(self.system_prompt))
    
    @staticmethod
    def _extract_placeholders(template: str) -> set:
        """Extract placeholder names from template."""
        formatter = Formatter()
        placeholders = set()
        
        for _, field_name, _, _ in formatter.parse(template):
            if field_name is not None:
                # Handle nested access like {data.field}
                base_name = field_name.split(".")[0].split("[")[0]
                if base_name:
                    placeholders.add(base_name)
        
        return placeholders
    
    @property
    def placeholders(self) -> List[str]:
        """Get list of placeholder names."""
        return sorted(self._placeholders)
    
    def render(
        self,
        data: Dict[str, Any],
        strict: bool = False,
    ) -> str:
        """
        Render template with data values.
        
        Args:
            data: Dictionary of values to substitute
            strict: If True, raise error for missing placeholders
        
        Returns:
            Rendered template string
        """
        # Merge with defaults
        values = {**self.default_values, **data}
        
        # Check for missing placeholders
        if strict:
            missing = self._placeholders - set(values.keys())
            if missing:
                raise ValueError(f"Missing values for placeholders: {missing}")
        
        # Render template
        try:
            rendered = self.template.format(**values)
        except KeyError as e:
            if strict:
                raise
            # Replace missing placeholders with empty string
            rendered = self.template
            for ph in self._placeholders:
                if ph not in values:
                    rendered = rendered.replace(f"{{{ph}}}", "")
                else:
                    rendered = rendered.replace(f"{{{ph}}}", str(values[ph]))
        
        return rendered
    
    def render_system(self, data: Dict[str, Any] = None) -> Optional[str]:
        """Render system prompt with data values."""
        if self.system_prompt is None:
            return None
        
        data = data or {}
        values = {**self.default_values, **data}
        
        try:
            return self.system_prompt.format(**values)
        except KeyError:
            # Return as-is if placeholders can't be filled
            return self.system_prompt
    
    def render_chat_messages(
        self,
        data: Dict[str, Any],
        tokenizer: Any = None,
    ) -> Union[str, List[Dict[str, str]]]:
        """
        Render as chat messages for chat models.
        
        Args:
            data: Dictionary of values to substitute
            tokenizer: Tokenizer with apply_chat_template method
        
        Returns:
            Either formatted string or list of message dicts
        """
        messages = []
        
        # Add system message if present
        system = self.render_system(data)
        if system:
            messages.append({"role": "system", "content": system})
        
        # Add user message
        user_content = self.render(data)
        messages.append({"role": "user", "content": user_content})
        
        # Apply chat template if tokenizer provided
        if tokenizer is not None and hasattr(tokenizer, "apply_chat_template"):
            return tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        
        return messages
    
    def __str__(self) -> str:
        """String representation."""
        lines = [f"PromptTemplate:"]
        if self.system_prompt:
            lines.append(f"  System: {self.system_prompt[:50]}...")
        lines.append(f"  Template: {self.template[:50]}...")
        lines.append(f"  Placeholders: {self.placeholders}")
        return "\n".join(lines)
    
    def __repr__(self) -> str:
        return f"PromptTemplate(placeholders={self.placeholders})"


class PromptBuilder:
    """
    Builder for creating complex prompts with multiple components.
    
    Example:
        builder = PromptBuilder()
        builder.add_system("You are a translator.")
        builder.add_instruction("Translate the following text to English.")
        builder.add_input("{source}")
        builder.add_output_prefix("Translation:")
        
        prompt = builder.build().render({"source": "مرحبا"})
    """
    
    def __init__(self):
        self._system: Optional[str] = None
        self._parts: List[str] = []
        self._output_prefix: Optional[str] = None
    
    def add_system(self, text: str) -> "PromptBuilder":
        """Add system prompt."""
        self._system = text
        return self
    
    def add_text(self, text: str) -> "PromptBuilder":
        """Add text to prompt."""
        self._parts.append(text)
        return self
    
    def add_instruction(self, text: str) -> "PromptBuilder":
        """Add instruction text."""
        self._parts.append(text)
        return self
    
    def add_input(self, text: str, label: str = None) -> "PromptBuilder":
        """Add input section."""
        if label:
            self._parts.append(f"{label}: {text}")
        else:
            self._parts.append(text)
        return self
    
    def add_output_prefix(self, text: str) -> "PromptBuilder":
        """Add output prefix (what comes before model's response)."""
        self._output_prefix = text
        return self
    
    def add_newline(self) -> "PromptBuilder":
        """Add newline."""
        self._parts.append("")
        return self
    
    def build(self) -> PromptTemplate:
        """Build the prompt template."""
        template = "\n".join(self._parts)
        
        if self._output_prefix:
            template = template + "\n" + self._output_prefix
        
        return PromptTemplate(
            template=template,
            system_prompt=self._system,
        )


# Pre-built templates for common tasks
TEMPLATES = {
    "translation": PromptTemplate(
        template="Translate the following text to {target_language}:\n\n{source}\n\nTranslation:",
        system_prompt="You are a professional translator. Translate accurately and naturally.",
    ),
    
    "translation_simple": PromptTemplate(
        template="Translate to English: {source}",
    ),
    
    "summarization": PromptTemplate(
        template="Summarize the following text:\n\n{text}\n\nSummary:",
        system_prompt="You are a helpful assistant that provides concise summaries.",
    ),
    
    "qa": PromptTemplate(
        template="Context: {context}\n\nQuestion: {question}\n\nAnswer:",
        system_prompt="Answer questions based on the provided context.",
    ),
    
    "classification": PromptTemplate(
        template="Classify the following text into one of these categories: {categories}\n\nText: {text}\n\nCategory:",
    ),
    
    "completion": PromptTemplate(
        template="{text}",
    ),
}


def get_template(name: str) -> PromptTemplate:
    """Get a pre-built template by name."""
    if name not in TEMPLATES:
        raise ValueError(f"Unknown template: {name}. Available: {list(TEMPLATES.keys())}")
    return TEMPLATES[name]
