#!/usr/bin/env python3
"""
LLM Inference CLI Tool
======================

A modular, extensible CLI for batch LLM inference using Unsloth or HuggingFace.

Features:
- Batch inference on datasets (TSV, CSV, JSONL, TXT)
- Customizable prompt templates with placeholders
- Multiple quantization options (4-bit, 8-bit, 16-bit)
- Configurable generation parameters
- Output parsing and saving
- Resume capability for long jobs
- Progress tracking

Usage:
    llm-batch --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
              --input data.tsv \
              --template prompt.md \
              --output results.tsv

    llm-batch --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
              --input data.tsv \
              --template "Translate to English: {source}" \
              --output results.jsonl \
              --max-tokens 256 \
              --batch-size 8
"""

import argparse
import sys
import os
from pathlib import Path

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))

from llm_batch.engine import InferenceEngine, BatchInferenceEngine
from llm_batch.config import InferenceConfig
from llm_batch import __version__


def parse_args():
    parser = argparse.ArgumentParser(
        description="LLM Inference CLI - Batch inference with customizable templates",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic NMT inference
  llm-batch --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
            --input sentences.tsv \\
            --template "Translate Arabic to English: {source}" \\
            --output translations.tsv

  # Using a template file with system prompt
  llm-batch --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
            --input data.tsv \\
            --template prompts/nmt_template.md \\
            --system-prompt prompts/system.txt \\
            --output results.jsonl

  # Summarization with custom parameters
  llm-batch --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \\
            --input articles.jsonl \\
            --template "Summarize: {text}" \\
            --output summaries.tsv \\
            --max-tokens 512 \\
            --temperature 0.3

  # List supported models
  llm-batch --list-models

  # Resume interrupted job
  llm-batch --resume results.jsonl.checkpoint
        """
    )
    
    # Model options
    model_group = parser.add_argument_group("Model Options")
    model_group.add_argument(
        "--model", "-m",
        help="Model name or path (HuggingFace hub or local)"
    )
    model_group.add_argument(
        "--quantization", "-q",
        choices=["4bit", "8bit", "16bit", "none"],
        default="4bit",
        help="Quantization level (default: 4bit)"
    )
    model_group.add_argument(
        "--backend",
        choices=["unsloth", "transformers", "auto"],
        default="auto",
        help="Inference backend (default: auto)"
    )
    model_group.add_argument(
        "--dtype",
        choices=["float16", "bfloat16", "float32"],
        default="float16",
        help="Data type for inference (default: float16)"
    )
    model_group.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="Maximum sequence length (default: 4096)"
    )
    
    # Input/Output options
    io_group = parser.add_argument_group("Input/Output Options")
    io_group.add_argument(
        "--input", "-i",
        help="Input file (TSV, CSV, JSONL, or TXT)"
    )
    io_group.add_argument(
        "--output", "-o",
        help="Output file (TSV, CSV, JSONL)"
    )
    io_group.add_argument(
        "--input-cols",
        help="Comma-separated column names for TSV/CSV (default: auto-detect)"
    )
    io_group.add_argument(
        "--output-cols",
        default="input,output",
        help="Comma-separated output column names (default: input,output)"
    )
    io_group.add_argument(
        "--delimiter",
        default="\t",
        help="Delimiter for TSV/CSV files (default: tab)"
    )
    
    # Template options
    template_group = parser.add_argument_group("Template Options")
    template_group.add_argument(
        "--template", "-t",
        help="Prompt template (string or path to .txt/.md file)"
    )
    template_group.add_argument(
        "--system-prompt", "-s",
        help="System prompt (string or path to file)"
    )
    template_group.add_argument(
        "--chat-template",
        help="Chat template name (e.g., chatml, llama-3, qwen)"
    )
    template_group.add_argument(
        "--no-chat-template",
        action="store_true",
        help="Disable chat template, use raw completion"
    )
    
    # Generation options
    gen_group = parser.add_argument_group("Generation Options")
    gen_group.add_argument(
        "--max-tokens", "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum new tokens to generate (default: 256)"
    )
    gen_group.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    gen_group.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Top-p sampling (default: 0.9)"
    )
    gen_group.add_argument(
        "--top-k",
        type=int,
        default=50,
        help="Top-k sampling (default: 50)"
    )
    gen_group.add_argument(
        "--num-beams",
        type=int,
        default=1,
        help="Beam search beams (default: 1, greedy/sampling)"
    )
    gen_group.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.1,
        help="Repetition penalty (default: 1.1)"
    )
    gen_group.add_argument(
        "--no-repeat-ngram",
        type=int,
        default=0,
        help="No repeat n-gram size (default: 0, disabled)"
    )
    gen_group.add_argument(
        "--do-sample",
        action="store_true",
        default=True,
        help="Enable sampling (default: True)"
    )
    gen_group.add_argument(
        "--no-sample",
        action="store_true",
        help="Disable sampling, use greedy decoding"
    )
    
    # Processing options
    proc_group = parser.add_argument_group("Processing Options")
    proc_group.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)"
    )
    proc_group.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Data loading workers (default: 0)"
    )
    proc_group.add_argument(
        "--limit", "-n",
        type=int,
        help="Limit number of samples to process"
    )
    proc_group.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip first N samples"
    )
    proc_group.add_argument(
        "--resume",
        help="Resume from checkpoint file"
    )
    proc_group.add_argument(
        "--checkpoint-every",
        type=int,
        default=100,
        help="Save checkpoint every N samples (default: 100)"
    )
    
    # Output processing
    output_group = parser.add_argument_group("Output Processing")
    output_group.add_argument(
        "--strip-output",
        action="store_true",
        default=True,
        help="Strip whitespace from output (default: True)"
    )
    output_group.add_argument(
        "--extract-pattern",
        help="Regex pattern to extract from output"
    )
    output_group.add_argument(
        "--stop-strings",
        help="Comma-separated stop strings"
    )
    output_group.add_argument(
        "--include-input",
        action="store_true",
        help="Include input columns in output"
    )
    output_group.add_argument(
        "--include-prompt",
        action="store_true",
        help="Include full prompt in output"
    )
    
    # Misc options
    misc_group = parser.add_argument_group("Miscellaneous")
    misc_group.add_argument(
        "--list-models",
        action="store_true",
        help="List recommended models"
    )
    misc_group.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without running"
    )
    misc_group.add_argument(
        "--verbose", "-v",
        action="count",
        default=0,
        help="Increase verbosity (-v, -vv, -vvv)"
    )
    misc_group.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output"
    )
    misc_group.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)"
    )
    misc_group.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu, cuda:0, etc.)"
    )
    misc_group.add_argument(
        "--config",
        help="Load config from YAML/JSON file"
    )
    misc_group.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}"
    )
    
    return parser.parse_args()


def list_models():
    """Print list of recommended models."""
    print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         RECOMMENDED MODELS                                    ║
╠══════════════════════════════════════════════════════════════════════════════╣
║                                                                              ║
║  🏆 BEST FOR ARABIC/MULTILINGUAL:                                            ║
║  ─────────────────────────────────                                           ║
║  • unsloth/Qwen2.5-7B-Instruct-bnb-4bit     (★★★★★ multilingual)            ║
║  • unsloth/Qwen2.5-3B-Instruct-bnb-4bit     (★★★★★ efficient)               ║
║  • unsloth/Qwen3-8B                          (★★★★★ latest)                  ║
║  • CohereForAI/aya-23-8B                     (★★★★★ translation)            ║
║  • inceptionai/jais-family-6p7b-chat         (★★★★★ Arabic-native)          ║
║                                                                              ║
║  ⚡ FAST & EFFICIENT:                                                         ║
║  ────────────────────                                                        ║
║  • unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit   (~3GB VRAM)                      ║
║  • unsloth/Llama-3.2-1B-Instruct-bnb-4bit   (~3GB VRAM)                      ║
║  • unsloth/gemma-3-4b-it-unsloth-bnb-4bit   (~4GB VRAM)                      ║
║                                                                              ║
║  🎯 GENERAL PURPOSE:                                                          ║
║  ───────────────────                                                         ║
║  • unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit                               ║
║  • unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit                               ║
║  • unsloth/phi-4-unsloth-bnb-4bit                                            ║
║                                                                              ║
║  📝 FOR SUMMARIZATION:                                                        ║
║  ─────────────────────                                                       ║
║  • unsloth/Qwen2.5-7B-Instruct-bnb-4bit                                      ║
║  • unsloth/Llama-3.3-70B-Instruct-bnb-4bit  (if you have VRAM)              ║
║                                                                              ║
║  Usage: llm-batch -m unsloth/Qwen2.5-7B-Instruct-bnb-4bit ...               ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """)


def main():
    args = parse_args()
    
    # Handle special commands
    if args.list_models:
        list_models()
        return 0
    
    # Validate required arguments
    if not args.model and not args.resume:
        print("Error: --model is required (or --resume to continue a job)")
        print("Use --list-models to see recommended models")
        return 1
    
    if not args.input and not args.resume:
        print("Error: --input is required")
        return 1
    
    if not args.template and not args.resume:
        print("Error: --template is required")
        return 1
    
    if not args.output and not args.resume:
        print("Error: --output is required")
        return 1
    
    # Handle --no-sample flag
    if args.no_sample:
        args.do_sample = False
    
    # Build config
    config = InferenceConfig.from_args(args)
    
    if args.verbose >= 2:
        print(f"Config:\n{config}")
    
    # Dry run - just show what would happen
    if args.dry_run:
        print("\n=== DRY RUN ===")
        print(f"Model: {config.model}")
        print(f"Input: {config.input_file} ({config.input_format})")
        print(f"Output: {config.output_file}")
        print(f"Template: {config.template[:100]}...")
        if config.system_prompt:
            print(f"System: {config.system_prompt[:100]}...")
        print(f"Samples: {config.limit or 'all'}")
        return 0
    
    # Run inference
    try:
        # Use batched engine for batch_size > 1
        if config.batch_size > 1:
            if not args.quiet:
                print(f"Using batched inference (batch_size={config.batch_size})")
            engine = BatchInferenceEngine(config)
        else:
            engine = InferenceEngine(config)
        engine.run()
        return 0
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        return 130
    except Exception as e:
        print(f"\nError: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
