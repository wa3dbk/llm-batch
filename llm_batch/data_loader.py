"""
Data loading utilities for LLM Inference CLI.

Supports:
- TSV (tab-separated values)
- CSV (comma-separated values)
- JSONL (JSON lines)
- TXT (plain text, one item per line)
"""

import csv
import json
from pathlib import Path
from typing import List, Dict, Any, Iterator, Optional, Union
from dataclasses import dataclass

from .utils import detect_format


@dataclass
class DataItem:
    """Single data item for inference."""
    index: int
    data: Dict[str, Any]
    raw_line: Optional[str] = None
    
    def __getitem__(self, key: str) -> Any:
        return self.data.get(key)
    
    def get(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)
    
    def keys(self) -> List[str]:
        return list(self.data.keys())


class DataLoader:
    """
    Loads data from various file formats for batch inference.
    
    Supports TSV, CSV, JSONL, and TXT formats with automatic detection.
    """
    
    def __init__(
        self,
        filepath: str,
        format: str = "auto",
        columns: Optional[List[str]] = None,
        delimiter: str = "\t",
        skip: int = 0,
        limit: Optional[int] = None,
    ):
        """
        Initialize data loader.
        
        Args:
            filepath: Path to input file
            format: File format ("auto", "tsv", "csv", "jsonl", "txt")
            columns: Column names (for TSV/CSV without header)
            delimiter: Delimiter for TSV/CSV
            skip: Number of items to skip
            limit: Maximum number of items to load
        """
        self.filepath = Path(filepath)
        self.format = format if format != "auto" else detect_format(str(self.filepath))
        self.columns = columns
        self.delimiter = delimiter
        self.skip = skip
        self.limit = limit
        
        self._data: List[DataItem] = []
        self._loaded = False
    
    def load(self) -> List[DataItem]:
        """Load all data from file."""
        if self._loaded:
            return self._data
        
        loader_map = {
            "tsv": self._load_tsv,
            "csv": self._load_csv,
            "jsonl": self._load_jsonl,
            "txt": self._load_txt,
        }
        
        loader = loader_map.get(self.format)
        if loader is None:
            raise ValueError(f"Unsupported format: {self.format}")
        
        self._data = list(loader())
        self._loaded = True
        
        return self._data
    
    def _load_tsv(self) -> Iterator[DataItem]:
        """Load TSV file."""
        return self._load_delimited("\t")
    
    def _load_csv(self) -> Iterator[DataItem]:
        """Load CSV file."""
        return self._load_delimited(",")
    
    def _load_delimited(self, delimiter: str) -> Iterator[DataItem]:
        """Load delimiter-separated file."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            # Peek at first line to detect if it's a header
            first_line = f.readline().strip()
            f.seek(0)
            
            reader = csv.reader(f, delimiter=delimiter)
            
            # Determine columns
            if self.columns:
                columns = self.columns
                # Skip header if it matches our columns
                first_row = next(reader)
                if first_row != columns:
                    # First row is data, not header
                    if self.skip == 0:
                        yield DataItem(
                            index=0,
                            data=dict(zip(columns, first_row)),
                            raw_line=delimiter.join(first_row),
                        )
            else:
                # Assume first row is header
                columns = next(reader)
                columns = [c.strip() for c in columns]
            
            # Load remaining rows
            idx = 1 if self.columns else 0
            count = 0
            
            for row in reader:
                if count < self.skip:
                    count += 1
                    idx += 1
                    continue
                
                if self.limit and (count - self.skip) >= self.limit:
                    break
                
                # Handle rows with fewer columns
                while len(row) < len(columns):
                    row.append("")
                
                yield DataItem(
                    index=idx,
                    data=dict(zip(columns, row)),
                    raw_line=delimiter.join(row),
                )
                
                idx += 1
                count += 1
    
    def _load_jsonl(self) -> Iterator[DataItem]:
        """Load JSONL file."""
        with open(self.filepath, "r", encoding="utf-8") as f:
            idx = 0
            count = 0
            
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                if count < self.skip:
                    count += 1
                    idx += 1
                    continue
                
                if self.limit and (count - self.skip) >= self.limit:
                    break
                
                data = json.loads(line)
                
                yield DataItem(
                    index=idx,
                    data=data,
                    raw_line=line,
                )
                
                idx += 1
                count += 1
    
    def _load_txt(self) -> Iterator[DataItem]:
        """Load plain text file (one item per line)."""
        # Determine column name
        col_name = self.columns[0] if self.columns else "text"
        
        with open(self.filepath, "r", encoding="utf-8") as f:
            idx = 0
            count = 0
            
            for line in f:
                line = line.rstrip("\n\r")
                
                if count < self.skip:
                    count += 1
                    idx += 1
                    continue
                
                if self.limit and (count - self.skip) >= self.limit:
                    break
                
                yield DataItem(
                    index=idx,
                    data={col_name: line},
                    raw_line=line,
                )
                
                idx += 1
                count += 1
    
    def __iter__(self) -> Iterator[DataItem]:
        """Iterate over data items."""
        if not self._loaded:
            self.load()
        return iter(self._data)
    
    def __len__(self) -> int:
        """Return number of items."""
        if not self._loaded:
            self.load()
        return len(self._data)
    
    def __getitem__(self, idx: int) -> DataItem:
        """Get item by index."""
        if not self._loaded:
            self.load()
        return self._data[idx]
    
    def get_columns(self) -> List[str]:
        """Get column names from loaded data."""
        if not self._loaded:
            self.load()
        if self._data:
            return self._data[0].keys()
        return []
    
    def preview(self, n: int = 3) -> str:
        """Preview first n items."""
        if not self._loaded:
            self.load()
        
        lines = [f"Data Preview ({len(self)} items total):"]
        lines.append(f"Columns: {self.get_columns()}")
        lines.append("-" * 50)
        
        for item in self._data[:n]:
            lines.append(f"[{item.index}] {item.data}")
        
        if len(self._data) > n:
            lines.append(f"... and {len(self._data) - n} more")
        
        return "\n".join(lines)


def load_data(
    filepath: str,
    format: str = "auto",
    columns: Optional[List[str]] = None,
    delimiter: str = "\t",
    skip: int = 0,
    limit: Optional[int] = None,
) -> DataLoader:
    """
    Convenience function to load data.
    
    Args:
        filepath: Path to input file
        format: File format ("auto", "tsv", "csv", "jsonl", "txt")
        columns: Column names (for TSV/CSV without header)
        delimiter: Delimiter for TSV/CSV
        skip: Number of items to skip
        limit: Maximum number of items to load
    
    Returns:
        DataLoader instance with loaded data
    """
    loader = DataLoader(
        filepath=filepath,
        format=format,
        columns=columns,
        delimiter=delimiter,
        skip=skip,
        limit=limit,
    )
    loader.load()
    return loader
