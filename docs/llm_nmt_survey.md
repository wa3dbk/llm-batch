# LLMs for Machine Translation: Paper Summary & Best Practices

A summary of [Jon, Bondok & Bojar (AbjadNLP 2026)](https://aclanthology.org/2026.abjadnlp-1.41/) and
broader guidelines for using LLMs for NMT — as direct translators and as
teachers for smaller encoder-decoder models.

---

## 1. Paper: "Current state of LLMs for Arabic dialectal machine translation"

**Authors:** Josef Jon, Rawan Bondok, Ondřej Bojar  
**Venue:** AbjadNLP 2026, ACL  
**Dataset:** MADAR — 16 Arabic dialects + MSA, both En→Ar and Ar→En  
**Code:** <https://github.com/cepin19/arabic_llms>

### 1.1 Models Evaluated (16 LLMs)

| Category | Models |
|----------|--------|
| **Arabic-specialized** | Jais-2-70B-Chat, Jais-2-8B-Chat, Nile-Chat-12B, c4ai-command-r7b-arabic |
| **Multilingual** | Aya Expanse 8B/32B, c4ai-command-r-08-2024, c4ai-command-r-v01, Command-A-Translate-08-2025 (111B), Gemma-3-4B-IT, Gemma-3-27B-IT, EuroLLM-9B-Instruct, Mistral-Small-3.2-24B-Instruct-2506, Qwen3-4B-Instruct-2507, Llama-3.3-70B-Instruct |
| **Commercial API** | GPT-4.1-mini, GPT-4.1-nano |

### 1.2 Prompts Used

**System prompt** (applied to all models):

```
You are a professional, very precise translator and a native speaker.
Translate inputs based on the instructions and always print out only the
text of the best possible translation, with no explanations.
Keep the same formatting (e.g. markup, lines, spacing) as the original.
Do not translate untranslatable parts of the input (URLs, code, and similar).
```

**Instruction prompt — dialect-specific** (En→Ar):

```
Translate the following text into {lang}, only print out the translation,
not add any explanations: {line}
```

Where `{lang}` is e.g. "Egyptian Arabic", "Tunisian Arabic", etc.

**Instruction prompt — general Arabic** (En→Ar):

Same template but with `{lang}` = "Arabic" (no dialect specified).

**Instruction prompt — Arabic to English** (Ar→En):

A single prompt without specifying source language or dialect.

### 1.3 Inference Setup

- Open-source models deployed via **VLLM** with default decoding parameters.
- Commercial models accessed via OpenAI API with defaults.
- **Post-processing:** Output cropped when **>5x longer** than source (in tokens
  or characters) — a sign of degenerate/repetitive generation.
- **Text normalization:** CAMeL Tools (Alef Maksura, Alef, Teh Marbuta, Hamza
  normalizations; Arabic numerals converted to Latin).

### 1.4 Key Findings

**Model rankings (En→Ar, ChrF on dialect-specific prompt):**

| Rank | Model | Size | Avg Rank | Wins/16 |
|------|-------|------|----------|---------|
| 1 | GPT-4.1-mini | N/A | 2.1 | 16 |
| 2 | **Jais-2-70B-Chat** | 70B | 2.9 | 14 |
| 3 | GPT-4.1-nano | N/A | 3.5 | 0 |
| 4 | gemma-3-27b-it | 27B | 5.3 | 0 |
| 5 | c4ai-command-r-08-2024 | 32B | 6.0 | 0 |
| 6 | aya-expanse-32b | 32B | 6.7 | 0 |
| 7 | command-a-translate-08-2025 | 111B | 7.1 | 2 |

**Overall BLEU + ChrF (all test sets, dialect-specific prompt):**

| Model | BLEU | ChrF |
|-------|------|------|
| Jais-2-70B-Chat | 25.4 | 51.8 |
| Jais-2-8B-Chat | 14.5 | 41.6 |
| GPT-4.1-mini | 15.0 | 44.8 |
| gemma-3-27b-it | 10.5 | 39.2 |
| aya-expanse-32b | 9.9 | 38.0 |
| Llama-3.3-70B-Instruct | 8.6 | 35.3 |
| Qwen3-4B-Instruct-2507 | 4.6 | 27.8 |

**Core takeaways:**

1. **Jais-2-70B-Chat is the best open-source model** for dialectal Arabic MT,
   with the highest accuracy _and_ dialectalness scores in manual evaluation
   (accuracy: 94.6, dialect: 98.5, mean: 96.5).

2. **Dialect-specific prompts substantially outperform general prompts** — most
   models can condition on dialect instructions and produce more appropriate
   dialectal output.

3. **Arabic-to-English is easier**: differences between models are smaller, and
   most produce acceptable translations. Jais-2-70B-Chat still leads.

4. **Many multilingual models default to MSA** even when explicitly prompted for
   a dialect. EuroLLM-9B-Instruct and Nile-Chat-12B are notable examples
   (Nile defaults to Egyptian regardless of target dialect).

5. **COMET is unsuitable** for dialectal Arabic evaluation — it penalizes
   correct dialectal translations and favors MSA, contradicting human judgment.
   BLEU and ChrF correlate better with human scores.

6. **Manual evaluation** (100 sentences, Egyptian Arabic) revealed that the
   best models (Jais-2-70B, GPT-4.1-mini) had very few errors, while weaker
   models showed major dialect mismatch (62.8% of EuroLLM-9B outputs were
   wrong dialect).

### 1.5 Limitations Noted by Authors

- Training data for most models is unknown — MADAR test set may have been seen
  during training.
- Prompts were not optimized per-model.
- Country names as dialect identifiers are imprecise (some dialects span
  multiple countries).
- Manual evaluation was small-scale (100 sentences, 1 dialect).

---

## 2. Best Practices for LLM-based Translation

Drawn from the survey by [Gain et al. (2025)](https://arxiv.org/abs/2504.01919),
"Bridging the Linguistic Divide: A Survey on Leveraging Large Language Models
for Machine Translation," and related work.

### 2.1 Prompting Strategies

**Prompt format matters.** The simplest effective format is:

```
[source_language]: {input}
[target_language]:
```

Or instruction-based:

```
Translate the following text into {target_language}, only print out the
translation, not add any explanations: {source}
```

**Key guidelines:**

| Strategy | Effect |
|----------|--------|
| Specify only source and target language | Best overall results ([Gain et al., 2025](https://arxiv.org/abs/2504.01919)) |
| Use dialect-specific language names | +2–14 ChrF improvement over generic "Arabic" ([Jon et al., 2026](https://aclanthology.org/2026.abjadnlp-1.41/)) |
| Add "only print out the translation, no explanations" | Suppresses verbose output |
| System prompt: "You are a professional translator" | Reduces hallucination and commentary |
| 3–5 few-shot examples | Significant gains; diminishing returns beyond 5 ([Gain et al., 2025](https://arxiv.org/abs/2504.01919)) |
| High-quality examples > random examples | One bad example can degrade quality significantly |
| Greedy / beam search decoding (no sampling) | More deterministic, scorable output for evaluation |

**Pivot prompting** (translating via English as intermediate) improves results
for distant language pairs (e.g. German→Chinese), but is unnecessary when
English is already one of the pair.

### 2.2 Suppressing Verbose / Noisy Output

This is one of the biggest practical challenges with using LLMs for batch
translation. The literature identifies several strategies:

1. **Instruction-level:** Explicit instructions like "only print out the
   translation, with no explanations" in both system prompt and user prompt.
   This is the approach used by Jon et al. (2026) and is the simplest fix.

2. **Length-ratio cropping:** Discard output that exceeds N× the source length.
   Jon et al. use 5× as the threshold. This catches degenerate repetitions.
   Implemented in `llm-batch` as `--max-length-ratio`.

3. **Stop strings:** Force generation to stop at newlines, `\n\n`, or other
   delimiters that signal the model is about to add commentary.

4. **Regex extraction:** Post-process output with a pattern like
   `Translation:\s*(.*)` to extract only the translation portion.

5. **Fine-tuning:** Even minimal SFT (32 examples) teaches the model to
   produce clean translation-only output and dramatically reduces parsing
   issues ([Gain et al., 2025](https://arxiv.org/abs/2504.01919)).

6. **Constrained decoding:** Enforce output grammar/format at the token level.
   Active research area but not yet mainstream for MT
   ([Geng et al., 2024](https://arxiv.org/abs/2403.06988)).

### 2.3 Recommended Decoding Settings for NMT

For producing scorable, reproducible translations (as opposed to creative
text), use deterministic decoding:

```bash
--no-sample --num-beams 4 --repetition-penalty 1.1 --max-length-ratio 5
```

- `--no-sample`: Disables stochastic sampling (greedy or beam search only).
- `--num-beams 4`: Beam search produces higher-quality translations than
  greedy for MT specifically.
- `--repetition-penalty 1.1`: Mild penalty to avoid degenerate loops.
- `--max-length-ratio 5`: Safety net for cropping runaway output.

---

## 3. LLMs as Teachers: Knowledge Distillation for NMT

Using an LLM to generate parallel data for training a smaller, faster
encoder-decoder model (e.g. MarianNMT / OPUS-MT) is an active research area.
There are two main approaches.

### 3.1 Sequence-Level Knowledge Distillation (Forward Translation)

The standard approach: use the LLM teacher to translate a monolingual corpus,
creating a synthetic parallel corpus, then train the student on it.

**Key references:**

- [Baradaran et al. (2025)](https://arxiv.org/abs/2505.14423) — "Scaling
  Low-Resource MT via Synthetic Data Generation with LLMs"
  - **Teacher:** GPT-4o
  - **Student:** Transformer-base (60.6M params), OPUS-MT, NLLB-1.3B, Llama-3.2-3B
  - **Method:** Forward-translated English Europarl into 7 low-resource
    languages via OpenAI Batch API
  - **Cost:** ~$5,000 for the full synthetic dataset
  - **Result:** Student transformer trained on synthetic data alone surpasses
    billion-parameter models (NLLB, Llama) for Basque, Scottish Gaelic,
    Icelandic, Georgian
  - **Fine-tuned OPUS-MT** trails GPT-4o by only 1–2 ChrF points despite being
    orders of magnitude smaller

- [Puduppully et al. (2024)](https://arxiv.org/abs/2404.13813) — "From LLM to
  NMT: Advancing Low-Resource Machine Translation with Claude"
  - **Teacher:** Claude 3 Opus
  - **Student:** Traditional NMT (encoder-decoder)
  - **Language:** Yoruba-English
  - **Result:** Student meets or surpasses NLLB-54B and Google Translate
  - **Cost:** ~$46 for creating the distillation dataset

**Best practices for synthetic data generation:**

| Guideline | Source |
|-----------|--------|
| Prioritize volume over perfection for truly low-resource pairs | [Baradaran et al., 2025](https://arxiv.org/abs/2505.14423) |
| Use encoder-decoder students (OPUS-MT, NLLB), not general LLMs | [Baradaran et al., 2025](https://arxiv.org/abs/2505.14423) |
| Fine-tune NLLB-200 — it benefits consistently across all directions | [Baradaran et al., 2025](https://arxiv.org/abs/2505.14423) |
| 50–60% of full dataset gives most of the gains; diminishing returns after | [Baradaran et al., 2025](https://arxiv.org/abs/2505.14423) |
| Back-translation (target→source) yields larger adequacy gains than forward | [Gain et al., 2025](https://arxiv.org/abs/2504.01919) |
| Filter generated data with language ID and length-ratio heuristics | [Baradaran et al., 2025](https://arxiv.org/abs/2505.14423) |
| Use Batch APIs (OpenAI, Anthropic) to reduce cost by ~50% | [Baradaran et al., 2025](https://arxiv.org/abs/2505.14423) |

### 3.2 Selective Knowledge Distillation (Error Patching)

Rather than distilling the entire translation, focus the LLM teacher on
correcting specific errors the student makes.

**Key reference:**

- [Li et al. (NAACL 2024)](https://aclanthology.org/2024.naacl-long.358/) —
  "MT-PATCHER: Selective and Extendable Knowledge Distillation from Large
  Language Models for Machine Translation"
  - The LLM teacher identifies translation errors in the student's output
  - It then synthesizes diverse contexts and potential error patterns
  - Fine-tuning the student on just ~10% of examples achieves results
    comparable to full sequence-level KD
  - The approach is more sample-efficient and can proactively address unseen
    error patterns

### 3.3 Practical Pipeline: LLM Teacher → Small NMT Student

A concrete workflow for using `llm-batch` in a teacher-student pipeline:

```
Step 1: Prepare monolingual data
  └─ Collect target-language text (e.g. Arabic news, social media)

Step 2: Generate synthetic parallel data with LLM teacher
  └─ llm-batch -m Jais-2-70B-Chat \
       --template nmt_ar2en \
       -i mono_arabic.tsv -o parallel_ar_en.tsv \
       --no-sample --num-beams 4 \
       --max-length-ratio 5 --include-input

Step 3: Filter and clean
  └─ Remove pairs where output is empty or suspiciously long/short
  └─ Run language ID filtering (e.g. fasttext lid.176)
  └─ Normalize text (CAMeL Tools for Arabic)

Step 4: Train student model
  └─ Fine-tune OPUS-MT or MarianNMT on the synthetic parallel data
  └─ Or fine-tune NLLB-200-distilled with the synthetic pairs

Step 5: Evaluate
  └─ Score with sacrebleu (BLEU + ChrF) against held-out references
  └─ Compare student vs. teacher on the same test set
```

---

## 4. Model Selection Guide

### For direct LLM translation (when latency/cost is acceptable):

| Use Case | Recommended Model | Why |
|----------|-------------------|-----|
| Arabic dialects (best quality) | Jais-2-70B-Chat | Top open-source for dialectal Arabic |
| Arabic dialects (lower VRAM) | Jais-2-8B-Chat | Good quality at 8B |
| General multilingual | aya-expanse-32b, gemma-3-27b-it | Strong across many languages |
| Fast / efficient | Qwen2.5-7B-Instruct (4-bit) | ~8GB VRAM, good multilingual |
| Commercial baseline | GPT-4.1-mini | Best overall quality |

### For teacher-student distillation:

| Component | Recommended | Why |
|-----------|-------------|-----|
| Teacher (quality) | GPT-4o, Claude, Jais-2-70B-Chat | Highest translation quality |
| Teacher (cost) | GPT-4o via Batch API, Claude | 50% cost reduction with batch |
| Student (multilingual) | NLLB-200-distilled-1.3B | Benefits consistently from fine-tuning |
| Student (bilingual) | OPUS-MT / MarianNMT | Small, fast, easy to deploy |
| Student (Arabic) | Helsinki-NLP/opus-mt-ar-en | Pre-trained Arabic-English |

---

## References

1. Jon, J., Bondok, R., & Bojar, O. (2026). [Current state of LLMs for Arabic dialectal machine translation](https://aclanthology.org/2026.abjadnlp-1.41/). _AbjadNLP 2026_.

2. Gain, B., Bandyopadhyay, D., & Ekbal, A. (2025). [Bridging the Linguistic Divide: A Survey on Leveraging Large Language Models for Machine Translation](https://arxiv.org/abs/2504.01919). _arXiv:2504.01919_.

3. Baradaran, R. et al. (2025). [Scaling Low-Resource MT via Synthetic Data Generation with LLMs](https://arxiv.org/abs/2505.14423). _arXiv:2505.14423_.

4. Puduppully, R. et al. (2024). [From LLM to NMT: Advancing Low-Resource Machine Translation with Claude](https://arxiv.org/abs/2404.13813). _arXiv:2404.13813_.

5. Li, J., Cheng, S., Huang, S., & Chen, J. (2024). [MT-PATCHER: Selective and Extendable Knowledge Distillation from Large Language Models for Machine Translation](https://aclanthology.org/2024.naacl-long.358/). _NAACL 2024_.

6. Kocmi, T. & Federmann, C. (2023). [GEMBA-MQM: Detecting translation quality error spans with GPT-4](https://aclanthology.org/2023.wmt-1.64/). _WMT 2023_.

7. Bouamor, H. et al. (2018). [The MADAR Arabic Dialect Corpus and Lexicon](https://aclanthology.org/L18-1535/). _LREC 2018_.

8. Geng, S. et al. (2024). [Guiding LLMs The Right Way: Fast, Non-Invasive Constrained Generation](https://arxiv.org/abs/2403.06988). _arXiv:2403.06988_.
