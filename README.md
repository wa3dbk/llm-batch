# LLM Inference CLI

A modular, extensible command-line tool for batch LLM inference with customizable prompt templates.

## Features

- 🚀 **Memory-efficient**: Uses Unsloth or 4-bit quantization for low VRAM usage
- 📊 **Batch processing**: Process TSV, CSV, JSONL, or plain text files
- 📝 **Flexible templates**: Use placeholders like `{column_name}` in prompts
- 💾 **Resume capability**: Checkpoint saves allow resuming interrupted jobs
- 🔧 **Extensible**: Modular design for easy customization

## Installation

```bash
# Create new environment
conda create -n llmbatch_env python=3.11 -y
conda activate llmbatch_env

# Clone the llm_batch repository
git clone https://github.com/wa3dbk/llm-batch.git

cd llm_batch

# Install in development mode
pip install -e .

# Or install dependencies manually
pip install torch transformers datasets accelerate peft trl bitsandbytes tqdm pyyaml

# Optional: Install Unsloth for 2x faster inference
pip install "unsloth[colab-new] @ git+https://github.com/unslothai/unsloth.git"

# Or the stable version
pip install unsloth
```

## Quick Start

### Basic Translation with batching

```bash
llm-batch \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --input sentences.tsv \
    --template "Translate to English: {source}" \
    --output translations.tsv \
    --batch-size 8 \
    --num-beams 4    
```

### Using Template Files

```bash
llm-batch \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --input data.tsv \
    --template templates/nmt_arabic_english.md \
    --system-prompt templates/system_translator.txt \
    --batch-size 8 \
    --output results.jsonl
```

### Summarization

```bash
llm-batch \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --input articles.jsonl \
    --template "Summarize in 3 sentences:\n\n{text}\n\nSummary:" \
    --output summaries.tsv \
    --max-tokens 256 \
    --temperature 0.3
```

## Input Formats

### TSV/CSV

Tab or comma-separated files with headers:

```
source	target
مرحبا	Hello
كيفاش حالك	How are you
```

### JSONL

JSON Lines format:

```json
{"source": "مرحبا", "id": 1}
{"source": "كيفاش حالك", "id": 2}
```

### Plain Text

One item per line:

```
مرحبا
كيفاش حالك
```

## Prompt Templates

Templates support `{placeholder}` syntax where placeholders are replaced with values from your input data.

### Inline Template

```bash
--template "Translate to English: {source}"
```

### Template File (Markdown)

Create `templates/my_template.md`:

```markdown
# Translation Task

Translate the following Arabic text to English.

## Arabic:
{source}

## English:
```

Then use it:

```bash
--template templates/my_template.md
```

### With System Prompt

```bash
--template "Translate: {source}" \
--system-prompt "You are a professional translator."
```

## Command-Line Options

### Model Options

| Option | Description | Default |
|--------|-------------|---------|
| `--model, -m` | Model name or path | Required |
| `--quantization, -q` | Quantization level (4bit, 8bit, 16bit, none) | 4bit |
| `--backend` | Backend (unsloth, transformers, auto) | auto |
| `--dtype` | Data type (float16, bfloat16, float32) | float16 |
| `--max-seq-len` | Maximum sequence length | 4096 |

### Input/Output Options

| Option | Description | Default |
|--------|-------------|---------|
| `--input, -i` | Input file path | Required |
| `--output, -o` | Output file path | Required |
| `--input-cols` | Column names (comma-separated) | Auto-detect |
| `--delimiter` | Delimiter for TSV/CSV | Tab |

### Template Options

| Option | Description | Default |
|--------|-------------|---------|
| `--template, -t` | Prompt template (string or file path) | Required |
| `--system-prompt, -s` | System prompt (string or file path) | None |
| `--no-chat-template` | Disable chat template formatting | False |

### Generation Options

| Option | Description | Default |
|--------|-------------|---------|
| `--max-tokens` | Maximum new tokens to generate | 256 |
| `--temperature` | Sampling temperature | 0.7 |
| `--top-p` | Top-p (nucleus) sampling | 0.9 |
| `--top-k` | Top-k sampling | 50 |
| `--num-beams` | Beam search beams (1 = no beam search) | 1 |
| `--repetition-penalty` | Repetition penalty | 1.1 |
| `--no-repeat-ngram` | No repeat n-gram size | 0 |
| `--no-sample` | Use greedy decoding | False |

### Processing Options

| Option | Description | Default |
|--------|-------------|---------|
| `--batch-size` | Batch size for inference | 1 |
| `--limit, -n` | Limit number of samples | None |
| `--skip` | Skip first N samples | 0 |
| `--checkpoint-every` | Save checkpoint every N items | 100 |
| `--resume` | Resume from checkpoint file | None |

### Output Processing

| Option | Description | Default |
|--------|-------------|---------|
| `--strip-output` | Strip whitespace from output | True |
| `--extract-pattern` | Regex pattern to extract from output | None |
| `--stop-strings` | Comma-separated stop strings | None |
| `--include-input` | Include input columns in output | False |
| `--include-prompt` | Include full prompt in output | False |

### Miscellaneous

| Option | Description | Default |
|--------|-------------|---------|
| `--list-models` | List recommended models | - |
| `--dry-run` | Show what would be processed | False |
| `--verbose, -v` | Increase verbosity | 0 |
| `--quiet` | Suppress progress output | False |
| `--seed` | Random seed | 42 |

## Recommended Models

### Best for Arabic/Multilingual

| Model | VRAM | Notes |
|-------|------|-------|
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | ~8GB | Excellent multilingual |
| `unsloth/Qwen2.5-3B-Instruct-bnb-4bit` | ~5GB | Good balance |
| `CohereForAI/aya-23-8B` | ~10GB | Great for translation |
| `inceptionai/jais-family-6p7b-chat` | ~8GB | Best Arabic tokenizer |

### Fast & Efficient

| Model | VRAM | Notes |
|-------|------|-------|
| `unsloth/Qwen2.5-1.5B-Instruct-bnb-4bit` | ~3GB | Fast, decent quality |
| `unsloth/Llama-3.2-1B-Instruct-bnb-4bit` | ~3GB | Very fast |
| `unsloth/gemma-3-4b-it-unsloth-bnb-4bit` | ~4GB | Good quality |

### High Quality

| Model | VRAM | Notes |
|-------|------|-------|
| `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` | ~10GB | Strong general |
| `unsloth/Mistral-Nemo-Instruct-2407-bnb-4bit` | ~14GB | High quality |

## Examples

### NMT Evaluation

```bash
# Prepare test data (test.tsv with 'source' and 'reference' columns)

# Run inference
llm-batch \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --input test.tsv \
    --template "Translate Arabic to English: {source}" \
    --output predictions.tsv \
    --num-beams 4 \
    --max-tokens 256

# Evaluate with sacrebleu
cut -f2 predictions.tsv | sacrebleu test.reference.txt
```

### Batch Summarization

```bash
llm-batch \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --input articles.jsonl \
    --template templates/summarization.md \
    --output summaries.jsonl \
    --max-tokens 512 \
    --temperature 0.3 \
    --include-input
```

### Resume Interrupted Job

```bash
# If job was interrupted, resume from checkpoint
llm-batch --resume results.jsonl.checkpoint
```

### Extract Specific Output

```bash
# Extract only the translation from verbose output
llm-batch \
    --model unsloth/Qwen2.5-7B-Instruct-bnb-4bit \
    --input data.tsv \
    --template "Translate: {source}\nTranslation:" \
    --output results.tsv \
    --extract-pattern "Translation:\s*(.*)"
```

## Python API

```python
from llm_batch import InferenceEngine, InferenceConfig

# Create config
config = InferenceConfig(
    model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    input_file="data.tsv",
    template="Translate to English: {source}",
    output_file="results.tsv",
    max_new_tokens=256,
    num_beams=4,
)

# Run inference
engine = InferenceEngine(config)
engine.run()
```

Or use the convenience function:

```python
from llm_batch import run_inference

run_inference(
    model="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    input_file="data.tsv",
    template="Translate to English: {source}",
    output_file="results.tsv",
)
```

## Architecture

```
llm_batch/
├── __init__.py      # Package exports
├── cli.py           # Command-line interface
├── config.py        # Configuration management
├── model_loader.py  # Model loading (Unsloth/HF)
├── data_loader.py   # Data loading (TSV/CSV/JSONL/TXT)
├── template.py      # Prompt template handling
├── output.py        # Output writing and processing
└── engine.py        # Main inference engine
```

## Troubleshooting

### Out of Memory

- Use `--quantization 4bit` (default)
- Reduce `--max-seq-len`
- Use a smaller model

### Slow Inference

- Install Unsloth: `pip install unsloth`
- Use `--num-beams 1` (disable beam search)
- Increase `--batch-size` (if VRAM allows)

### Repetitive Output

- Increase `--repetition-penalty` (e.g., 1.2)
- Add `--no-repeat-ngram 3`
- Lower `--temperature`

### Import Errors

```bash
# If Unsloth has issues, force transformers backend
llm-batch --backend transformers ...
```

## License

MIT License
