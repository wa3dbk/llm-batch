"""
Output handling for LLM Inference CLI.

Supports:
- TSV output
- CSV output  
- JSONL output
- Checkpointing for resume capability
- Output post-processing (strip, extract patterns)
"""

import csv
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class InferenceResult:
    """Single inference result."""
    index: int
    input_data: Dict[str, Any]
    prompt: str
    output: str
    raw_output: str
    metadata: Optional[Dict[str, Any]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "index": self.index,
            "input": self.input_data,
            "prompt": self.prompt,
            "output": self.output,
            "raw_output": self.raw_output,
            "metadata": self.metadata,
        }


class OutputProcessor:
    """Post-processes model outputs."""
    
    def __init__(
        self,
        strip: bool = True,
        extract_pattern: Optional[str] = None,
        stop_strings: Optional[List[str]] = None,
    ):
        """
        Initialize output processor.
        
        Args:
            strip: Strip whitespace from output
            extract_pattern: Regex pattern to extract from output
            stop_strings: Strings to stop generation at
        """
        self.strip = strip
        self.extract_pattern = re.compile(extract_pattern) if extract_pattern else None
        self.stop_strings = stop_strings or []
    
    def process(self, output: str) -> str:
        """Process model output."""
        processed = output
        
        # Strip whitespace
        if self.strip:
            processed = processed.strip()
        
        # Stop at stop strings
        for stop in self.stop_strings:
            if stop in processed:
                processed = processed.split(stop)[0]
        
        # Extract pattern
        if self.extract_pattern:
            match = self.extract_pattern.search(processed)
            if match:
                # Return first group if available, else full match
                processed = match.group(1) if match.groups() else match.group(0)
        
        # Final strip
        if self.strip:
            processed = processed.strip()
        
        return processed


class OutputWriter:
    """
    Writes inference results to various formats.
    
    Supports TSV, CSV, JSONL with automatic checkpointing.
    """
    
    def __init__(
        self,
        filepath: str,
        format: str = "auto",
        columns: Optional[List[str]] = None,
        delimiter: str = "\t",
        include_input: bool = False,
        include_prompt: bool = False,
        checkpoint_every: int = 100,
    ):
        """
        Initialize output writer.
        
        Args:
            filepath: Output file path
            format: Output format ("auto", "tsv", "csv", "jsonl")
            columns: Column names for output
            delimiter: Delimiter for TSV/CSV
            include_input: Include input data in output
            include_prompt: Include full prompt in output
            checkpoint_every: Save checkpoint every N items
        """
        self.filepath = Path(filepath)
        self.format = format if format != "auto" else self._detect_format()
        self.columns = columns
        self.delimiter = delimiter
        self.include_input = include_input
        self.include_prompt = include_prompt
        self.checkpoint_every = checkpoint_every
        
        self._results: List[InferenceResult] = []
        self._file = None
        self._writer = None
        self._header_written = False
    
    def _detect_format(self) -> str:
        """Detect format from file extension."""
        ext = self.filepath.suffix.lower()
        format_map = {
            ".tsv": "tsv",
            ".csv": "csv",
            ".jsonl": "jsonl",
            ".json": "jsonl",
        }
        return format_map.get(ext, "tsv")
    
    def open(self):
        """Open output file for writing."""
        self.filepath.parent.mkdir(parents=True, exist_ok=True)
        self._file = open(self.filepath, "w", encoding="utf-8", newline="")
        
        if self.format in ["tsv", "csv"]:
            delimiter = self.delimiter if self.format == "tsv" else ","
            self._writer = csv.writer(self._file, delimiter=delimiter)
    
    def close(self):
        """Close output file."""
        if self._file:
            self._file.close()
            self._file = None
            self._writer = None
    
    def __enter__(self):
        self.open()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
    
    def write(self, result: InferenceResult):
        """Write a single result."""
        self._results.append(result)
        
        if self.format == "jsonl":
            self._write_jsonl(result)
        else:
            self._write_delimited(result)
        
        # Checkpoint
        if len(self._results) % self.checkpoint_every == 0:
            self._save_checkpoint()
    
    def _write_jsonl(self, result: InferenceResult):
        """Write result as JSONL."""
        data = {"output": result.output}
        
        if self.include_input:
            data["input"] = result.input_data
        
        if self.include_prompt:
            data["prompt"] = result.prompt
        
        data["index"] = result.index
        
        self._file.write(json.dumps(data, ensure_ascii=False) + "\n")
        self._file.flush()
    
    def _write_delimited(self, result: InferenceResult):
        """Write result as TSV/CSV."""
        # Determine columns on first write
        if not self._header_written:
            if self.columns is None:
                self.columns = []
                if self.include_input:
                    self.columns.extend(result.input_data.keys())
                if self.include_prompt:
                    self.columns.append("prompt")
                self.columns.append("output")
            
            self._writer.writerow(self.columns)
            self._header_written = True
        
        # Build row
        row = []
        for col in self.columns:
            if col == "output":
                row.append(result.output)
            elif col == "prompt":
                row.append(result.prompt)
            elif col in result.input_data:
                row.append(result.input_data[col])
            else:
                row.append("")
        
        self._writer.writerow(row)
        self._file.flush()
    
    def _save_checkpoint(self):
        """Save checkpoint for resume capability."""
        checkpoint_path = Path(str(self.filepath) + ".checkpoint")
        
        checkpoint_data = {
            "count": len(self._results),
            "last_index": self._results[-1].index if self._results else -1,
            "filepath": str(self.filepath),
        }
        
        checkpoint_path.write_text(
            json.dumps(checkpoint_data, indent=2),
            encoding="utf-8"
        )
    
    def write_all(self, results: List[InferenceResult]):
        """Write all results."""
        for result in results:
            self.write(result)
    
    @staticmethod
    def load_checkpoint(checkpoint_path: str) -> Dict[str, Any]:
        """Load checkpoint data."""
        path = Path(checkpoint_path)
        if not path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        return json.loads(path.read_text(encoding="utf-8"))
    
    @property
    def count(self) -> int:
        """Number of results written."""
        return len(self._results)


class ResultCollector:
    """
    Collects and aggregates inference results.
    
    Useful for computing metrics after inference.
    """
    
    def __init__(self):
        self._results: List[InferenceResult] = []
    
    def add(self, result: InferenceResult):
        """Add a result."""
        self._results.append(result)
    
    def get_outputs(self) -> List[str]:
        """Get all outputs."""
        return [r.output for r in self._results]
    
    def get_inputs(self, key: str) -> List[Any]:
        """Get input values for a specific key."""
        return [r.input_data.get(key) for r in self._results]
    
    def to_dataframe(self):
        """Convert to pandas DataFrame."""
        import pandas as pd
        
        rows = []
        for r in self._results:
            row = {"index": r.index, "output": r.output}
            row.update(r.input_data)
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def __len__(self) -> int:
        return len(self._results)
    
    def __iter__(self):
        return iter(self._results)
