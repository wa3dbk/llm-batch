"""
Configuration management for LLM Inference CLI.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any
from pathlib import Path
import json
import yaml


@dataclass
class InferenceConfig:
    """Configuration for inference runs."""
    
    # Model settings
    model: str = ""
    quantization: str = "4bit"  # 4bit, 8bit, 16bit, none
    backend: str = "auto"  # unsloth, transformers, auto
    dtype: str = "float16"
    max_seq_len: int = 4096
    device: str = "auto"
    
    # Input/Output
    input_file: str = ""
    output_file: str = ""
    input_format: str = "auto"  # auto, tsv, csv, jsonl, txt
    output_format: str = "auto"
    input_cols: Optional[List[str]] = None
    output_cols: List[str] = field(default_factory=lambda: ["input", "output"])
    delimiter: str = "\t"
    
    # Template
    template: str = ""
    system_prompt: Optional[str] = None
    chat_template: Optional[str] = None
    use_chat_template: bool = True
    
    # Generation parameters
    max_new_tokens: int = 256
    temperature: float = 0.7
    top_p: float = 0.9
    top_k: int = 50
    num_beams: int = 1
    repetition_penalty: float = 1.1
    no_repeat_ngram_size: int = 0
    do_sample: bool = True
    stop_strings: Optional[List[str]] = None
    
    # Processing
    batch_size: int = 1
    num_workers: int = 0
    limit: Optional[int] = None
    skip: int = 0
    checkpoint_every: int = 100
    resume_from: Optional[str] = None
    
    # Output processing
    strip_output: bool = True
    extract_pattern: Optional[str] = None
    include_input: bool = False
    include_prompt: bool = False
    
    # Misc
    seed: int = 42
    verbose: int = 0
    quiet: bool = False
    
    def __post_init__(self):
        """Validate and process config after initialization."""
        # Auto-detect input format
        if self.input_format == "auto" and self.input_file:
            self.input_format = self._detect_format(self.input_file)
        
        # Auto-detect output format
        if self.output_format == "auto" and self.output_file:
            self.output_format = self._detect_format(self.output_file)
        
        # Parse stop strings
        if isinstance(self.stop_strings, str):
            self.stop_strings = [s.strip() for s in self.stop_strings.split(",")]
        
        # Load template from file if path exists
        if self.template and Path(self.template).exists():
            self.template = Path(self.template).read_text(encoding="utf-8")
        
        # Load system prompt from file if path exists
        if self.system_prompt and Path(self.system_prompt).exists():
            self.system_prompt = Path(self.system_prompt).read_text(encoding="utf-8")
    
    @staticmethod
    def _detect_format(filepath: str) -> str:
        """Detect file format from extension."""
        ext = Path(filepath).suffix.lower()
        format_map = {
            ".tsv": "tsv",
            ".csv": "csv",
            ".jsonl": "jsonl",
            ".json": "jsonl",
            ".txt": "txt",
        }
        return format_map.get(ext, "tsv")
    
    @classmethod
    def from_args(cls, args) -> "InferenceConfig":
        """Create config from argparse namespace."""
        config = cls(
            # Model
            model=args.model or "",
            quantization=args.quantization,
            backend=args.backend,
            dtype=args.dtype,
            max_seq_len=args.max_seq_len,
            device=args.device,
            
            # I/O
            input_file=args.input or "",
            output_file=args.output or "",
            input_cols=args.input_cols.split(",") if args.input_cols else None,
            output_cols=args.output_cols.split(","),
            delimiter=args.delimiter,
            
            # Template
            template=args.template or "",
            system_prompt=args.system_prompt,
            chat_template=args.chat_template,
            use_chat_template=not args.no_chat_template,
            
            # Generation
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            num_beams=args.num_beams,
            repetition_penalty=args.repetition_penalty,
            no_repeat_ngram_size=args.no_repeat_ngram,
            do_sample=args.do_sample,
            stop_strings=args.stop_strings.split(",") if args.stop_strings else None,
            
            # Processing
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            limit=args.limit,
            skip=args.skip,
            checkpoint_every=args.checkpoint_every,
            resume_from=args.resume,
            
            # Output
            strip_output=args.strip_output,
            extract_pattern=args.extract_pattern,
            include_input=args.include_input,
            include_prompt=args.include_prompt,
            
            # Misc
            seed=args.seed,
            verbose=args.verbose,
            quiet=args.quiet,
        )
        return config
    
    @classmethod
    def from_file(cls, filepath: str) -> "InferenceConfig":
        """Load config from YAML or JSON file."""
        path = Path(filepath)
        content = path.read_text(encoding="utf-8")
        
        if path.suffix in [".yaml", ".yml"]:
            data = yaml.safe_load(content)
        else:
            data = json.loads(content)
        
        return cls(**data)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith("_")
        }
    
    def save(self, filepath: str):
        """Save config to file."""
        path = Path(filepath)
        data = self.to_dict()
        
        if path.suffix in [".yaml", ".yml"]:
            content = yaml.dump(data, default_flow_style=False)
        else:
            content = json.dumps(data, indent=2)
        
        path.write_text(content, encoding="utf-8")
    
    def __str__(self) -> str:
        """Pretty print config."""
        lines = ["InferenceConfig:"]
        for k, v in self.to_dict().items():
            if v is not None and v != "":
                lines.append(f"  {k}: {v}")
        return "\n".join(lines)
