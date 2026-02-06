"""
Model loading utilities for LLM Inference CLI.

Supports:
- Unsloth (memory-efficient, fast)
- HuggingFace Transformers (fallback)
- Various quantization options (4-bit, 8-bit, 16-bit)
"""

import logging
import os

import torch
from typing import Tuple, Optional, Any
from pathlib import Path

logger = logging.getLogger(__name__)

# Disable Unsloth statistics to avoid vLLM errors
os.environ.setdefault("UNSLOTH_DISABLE_STATISTICS", "1")


class ModelLoader:
    """Handles model loading with Unsloth or HuggingFace."""
    
    # Models known to work well with Unsloth
    UNSLOTH_SUPPORTED_PREFIXES = [
        "unsloth/",
        "Qwen/",
        "meta-llama/",
        "mistralai/",
        "google/gemma",
        "microsoft/phi",
    ]
    
    def __init__(
        self,
        model_name: str,
        quantization: str = "4bit",
        backend: str = "auto",
        dtype: str = "float16",
        max_seq_len: int = 4096,
        device: str = "auto",
        verbose: int = 0,
    ):
        self.model_name = model_name
        self.quantization = quantization
        self.backend = backend
        self.dtype = dtype
        self.max_seq_len = max_seq_len
        self.device = device
        self.verbose = verbose
        
        self.model = None
        self.tokenizer = None
        self._backend_used = None
    
    def load(self) -> Tuple[Any, Any]:
        """Load model and tokenizer."""
        
        # Determine backend
        if self.backend == "auto":
            self._backend_used = self._detect_backend()
        else:
            self._backend_used = self.backend
        
        if self.verbose:
            logger.debug("Loading model: %s", self.model_name)
            logger.debug("Backend: %s", self._backend_used)
            logger.debug("Quantization: %s", self.quantization)
        
        if self._backend_used == "unsloth":
            return self._load_unsloth()
        else:
            return self._load_transformers()
    
    def _detect_backend(self) -> str:
        """Auto-detect best backend for model."""
        # Check if unsloth is available
        try:
            import unsloth
            unsloth_available = True
        except ImportError:
            unsloth_available = False
        
        if not unsloth_available:
            return "transformers"
        
        # Check if model is known to work with Unsloth
        for prefix in self.UNSLOTH_SUPPORTED_PREFIXES:
            if self.model_name.startswith(prefix):
                return "unsloth"
        
        # Default to transformers for unknown models
        return "transformers"
    
    def _get_torch_dtype(self):
        """Convert dtype string to torch dtype."""
        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        return dtype_map.get(self.dtype, torch.float16)
    
    def _load_unsloth(self) -> Tuple[Any, Any]:
        """Load model using Unsloth."""
        from unsloth import FastLanguageModel
        
        load_in_4bit = self.quantization == "4bit"
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=self.model_name,
            max_seq_length=self.max_seq_len,
            dtype=self._get_torch_dtype(),
            load_in_4bit=load_in_4bit,
        )
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        # Ensure pad token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    def _load_transformers(self) -> Tuple[Any, Any]:
        """Load model using HuggingFace Transformers."""
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        
        # Quantization config
        bnb_config = None
        if self.quantization == "4bit":
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=self._get_torch_dtype(),
                bnb_4bit_use_double_quant=True,
            )
        elif self.quantization == "8bit":
            bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="left",
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Device map
        device_map = "auto"
        if self.device != "auto":
            device_map = self.device
        
        # Load model
        model_kwargs = {
            "trust_remote_code": True,
            "device_map": device_map,
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config
        else:
            model_kwargs["torch_dtype"] = self._get_torch_dtype()
        
        # Use Flash Attention 2 if available
        try:
            import flash_attn  # noqa: F401
            model_kwargs["attn_implementation"] = "flash_attention_2"
            logger.debug("Flash Attention 2 enabled")
        except ImportError:
            logger.debug("Flash Attention 2 not available, using default attention")

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        model.eval()
        
        self.model = model
        self.tokenizer = tokenizer
        
        return model, tokenizer
    
    @property
    def backend_used(self) -> str:
        """Return the backend actually used."""
        return self._backend_used or "unknown"
    
    def get_model_info(self) -> dict:
        """Get information about loaded model."""
        info = {
            "name": self.model_name,
            "backend": self.backend_used,
            "quantization": self.quantization,
            "dtype": self.dtype,
            "max_seq_len": self.max_seq_len,
        }
        
        if self.model is not None:
            # Count parameters
            total_params = sum(p.numel() for p in self.model.parameters())
            info["total_params"] = total_params
            info["total_params_human"] = f"{total_params / 1e9:.2f}B"
        
        if self.tokenizer is not None:
            info["vocab_size"] = len(self.tokenizer)
            info["eos_token"] = repr(self.tokenizer.eos_token)
            info["pad_token"] = repr(self.tokenizer.pad_token)
        
        return info


def load_model(
    model_name: str,
    quantization: str = "4bit",
    backend: str = "auto",
    dtype: str = "float16",
    max_seq_len: int = 4096,
    device: str = "auto",
    verbose: int = 0,
) -> Tuple[Any, Any]:
    """
    Convenience function to load model and tokenizer.
    
    Args:
        model_name: HuggingFace model name or local path
        quantization: "4bit", "8bit", "16bit", or "none"
        backend: "unsloth", "transformers", or "auto"
        dtype: "float16", "bfloat16", or "float32"
        max_seq_len: Maximum sequence length
        device: Device to use ("auto", "cuda", "cpu", etc.)
        verbose: Verbosity level
    
    Returns:
        Tuple of (model, tokenizer)
    """
    loader = ModelLoader(
        model_name=model_name,
        quantization=quantization,
        backend=backend,
        dtype=dtype,
        max_seq_len=max_seq_len,
        device=device,
        verbose=verbose,
    )
    return loader.load()
