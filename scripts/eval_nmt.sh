#!/usr/bin/env bash
# =============================================================================
# NMT Evaluation Script
# =============================================================================
#
# Runs multiple LLMs on the same test set and produces parallel output files
# that can be scored with sacrebleu / chrF.
#
# Usage:
#   bash scripts/eval_nmt.sh
#
# Prerequisites:
#   pip install -e ".[eval]"     # sacrebleu, pandas
#   pip install unsloth          # optional, for Unsloth models
#
# After running, score with:
#   for f in eval_output/*_ar2en.tsv; do
#     echo "=== $(basename $f) ==="
#     tail -n +2 "$f" | cut -f1 | sacrebleu eval_data/reference_en.txt
#   done
# =============================================================================

set -euo pipefail

EVAL_DIR="eval_data"
OUT_DIR="eval_output"
mkdir -p "$EVAL_DIR" "$OUT_DIR"

# ---- Create test data if it doesn't exist ----

if [ ! -f "$EVAL_DIR/test_ar2en.tsv" ]; then
  echo "Creating Arabic-to-English test set..."
  cat > "$EVAL_DIR/test_ar2en.tsv" << 'EOF'
source
هذه أول جملة في النص
هو عاوز إيه بقى
لماذا لا تهتم بشكلي؟
كيفاش حالك اليوم؟
أنا رايح للسوق
ذهب الرجل إلى المقهى ومكث هناك إلى العصر
شنو هذا الشي؟
الجو حار بزاف اليوم
قولي له كاش
يعني سنة الماضية أخذوا رترات
يعني بتبلش معنا بالصيفية
وين مشيتي البارح؟
عطيني الكتاب هذاك
ما نحبش الماكلة هاذي
EOF
fi

if [ ! -f "$EVAL_DIR/reference_en.txt" ]; then
  echo "Creating English reference translations..."
  cat > "$EVAL_DIR/reference_en.txt" << 'EOF'
This is the first sentence.
Translate this, please.
Why don't you care about my appearance?
How are you today?
I'm going to the market.
What is this thing?
The weather is very hot today.
Where did you go yesterday?
Give me that book.
I don't like this food.
EOF
fi

if [ ! -f "$EVAL_DIR/test_en2ar.tsv" ]; then
  echo "Creating English-to-Arabic (dialect) test set..."
  cat > "$EVAL_DIR/test_en2ar.tsv" << 'EOF'
source	target_language
How are you today?	Tunisian Arabic
I'm going to the market.	Tunisian Arabic
The weather is very hot today.	Egyptian Arabic
Where did you go yesterday?	Egyptian Arabic
Give me that book.	Moroccan Arabic
I don't like this food.	Moroccan Arabic
This is the first sentence.	Iraqi Arabic
What is this thing?	Iraqi Arabic
Translate this please.	Lebanese Arabic
Why don't you care about my appearance?	Lebanese Arabic
EOF
fi

# ---- Model list ----
# Add or remove models depending on your available VRAM.
# Each entry: "model_id short_name"

MODELS_AR2EN=(
  "unsloth/Qwen2.5-7B-Instruct-bnb-4bit    qwen25_7b"
  "unsloth/Qwen2.5-3B-Instruct-bnb-4bit    qwen25_3b"
  "CohereForAI/aya-23-8B                    aya23_8b"
  "unsloth/gemma-3-4b-it-unsloth-bnb-4bit   gemma3_4b"
)

# Uncomment these if you have enough VRAM (>= 40GB):
# MODELS_AR2EN+=(
#   "inceptionai/jais-2-70b-chat              jais2_70b"
#   "Llama-3.3-70B-Instruct                   llama33_70b"
# )

# ---- Prompt variants to compare ----

declare -A PROMPTS
PROMPTS[simple]="Translate to English: {source}"
PROMPTS[nmt_builtin]="nmt_ar2en"              # built-in template with system prompt
PROMPTS[nmt_file]="templates/nmt_ar2en.md"     # file-based template

# ---- Shared flags ----
# --first-line-only strips multi-line commentary (Qwen verbose output)
# --max-length-ratio 5 crops degenerate repetitions (Jon et al. heuristic)

COMMON_FLAGS="--no-sample --max-length-ratio 5 --first-line-only --max-tokens 256 --quiet"

# =============================================================================
# Part 1: Arabic → English (all models, all prompt variants)
# =============================================================================

echo ""
echo "============================================================"
echo "  Part 1: Arabic → English"
echo "============================================================"

for entry in "${MODELS_AR2EN[@]}"; do
  model_id=$(echo "$entry" | awk '{print $1}')
  short_name=$(echo "$entry" | awk '{print $2}')

  for prompt_name in "${!PROMPTS[@]}"; do
    prompt_val="${PROMPTS[$prompt_name]}"
    outfile="$OUT_DIR/${short_name}_${prompt_name}_ar2en.tsv"

    if [ -f "$outfile" ]; then
      echo "[SKIP] $outfile already exists"
      continue
    fi

    echo ""
    echo ">>> Model: $model_id | Prompt: $prompt_name"
    echo "    Output: $outfile"

    # Determine template flags
    TEMPLATE_FLAGS=""
    if [ "$prompt_name" = "nmt_builtin" ]; then
      TEMPLATE_FLAGS="--template $prompt_val"
    elif [ "$prompt_name" = "nmt_file" ]; then
      TEMPLATE_FLAGS="--template $prompt_val --system-prompt templates/system_nmt.txt"
    else
      TEMPLATE_FLAGS="--template \"$prompt_val\""
    fi

    eval llm-batch \
      --model "$model_id" \
      --input "$EVAL_DIR/test_ar2en.tsv" \
      $TEMPLATE_FLAGS \
      --output "$outfile" \
      $COMMON_FLAGS \
      || echo "    [FAILED] $model_id / $prompt_name"
  done
done

# =============================================================================
# Part 2: English → Arabic dialects (subset of models)
# =============================================================================

echo ""
echo "============================================================"
echo "  Part 2: English → Arabic (dialect-specific)"
echo "============================================================"

MODELS_EN2AR=(
  "unsloth/Qwen2.5-7B-Instruct-bnb-4bit    qwen25_7b"
  "CohereForAI/aya-23-8B                    aya23_8b"
)

for entry in "${MODELS_EN2AR[@]}"; do
  model_id=$(echo "$entry" | awk '{print $1}')
  short_name=$(echo "$entry" | awk '{print $2}')
  outfile="$OUT_DIR/${short_name}_nmt_dialect_en2ar.tsv"

  if [ -f "$outfile" ]; then
    echo "[SKIP] $outfile already exists"
    continue
  fi

  echo ""
  echo ">>> Model: $model_id | Template: nmt (dialect-specific)"
  echo "    Output: $outfile"

  llm-batch \
    --model "$model_id" \
    --input "$EVAL_DIR/test_en2ar.tsv" \
    --template nmt \
    --output "$outfile" \
    --include-input \
    $COMMON_FLAGS \
    || echo "    [FAILED] $model_id / nmt dialect"
done

# =============================================================================
# Part 3: Score results (if sacrebleu is installed)
# =============================================================================

echo ""
echo "============================================================"
echo "  Scoring (Arabic → English)"
echo "============================================================"

if command -v sacrebleu &> /dev/null; then
  REF="$EVAL_DIR/reference_en.txt"
  echo ""
  printf "%-50s %8s %8s\n" "Output File" "BLEU" "ChrF"
  printf "%-50s %8s %8s\n" "----------" "----" "----"

  for f in "$OUT_DIR"/*_ar2en.tsv; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .tsv)

    # Extract the output column (skip header)
    hyp_file=$(mktemp)
    tail -n +2 "$f" | cut -f1 > "$hyp_file"

    # Check line count matches
    hyp_lines=$(wc -l < "$hyp_file" | tr -d ' ')
    ref_lines=$(wc -l < "$REF" | tr -d ' ')
    if [ "$hyp_lines" != "$ref_lines" ]; then
      printf "%-50s %8s %8s\n" "$name" "SKIP" "lines:$hyp_lines/$ref_lines"
      rm -f "$hyp_file"
      continue
    fi

    bleu=$(sacrebleu "$REF" -i "$hyp_file" -m bleu -b 2>/dev/null || echo "ERR")
    chrf=$(sacrebleu "$REF" -i "$hyp_file" -m chrf -b 2>/dev/null || echo "ERR")
    printf "%-50s %8s %8s\n" "$name" "$bleu" "$chrf"

    rm -f "$hyp_file"
  done
else
  echo "sacrebleu not installed. Install with: pip install sacrebleu"
  echo "Then score manually:"
  echo '  for f in eval_output/*_ar2en.tsv; do'
  echo '    echo "=== $(basename $f) ==="'
  echo '    tail -n +2 "$f" | cut -f1 | sacrebleu eval_data/reference_en.txt'
  echo '  done'
fi

echo ""
echo "Done. Output files are in $OUT_DIR/"
echo "To re-run a specific model, delete its output file first."
