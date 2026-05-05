# CARTS empirical explorations

This directory contains the empirical notebook and supporting utilities for a
paper on CARTS: Contextual Autoregressive Rank Transcoding Steganography.

The notebook runs six core experiments:

1. Implementation correctness
2. Rank-likelihood, rank-trace statistics, and simple detection
3. Key-collision fiber census and finite key search
4. Collision stability across new payloads
5. Non-commutativity of key-induced maps
6. Robustness to token perturbations

The experiments are finite empirical checks under one model, tokenizer,
payload sample, key sample, and comparison distribution. They do not prove
steganographic security or insecurity.

## Files

- `01_paper_empirical_experiments_llama8b.ipynb`: research notebook for the six
  experiments.
- `carts_empirical_utils.py`: token-id-level CARTS primitives, metrics, cache,
  plotting helpers, and summary writers.
- `results/figures/`: generated PNG figures.
- `results/tables/`: generated CSV tables and manifests.
- `results/text/`: generated Markdown summaries.
- `results/raw/`: raw JSON or JSONL result files.
- `results/cache/`: disk cache for expensive `F_k(r)` computations.

## Required model

The notebook defaults to Llama 3 8B:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

If this file is missing, the loader raises a clear error and stops. It does not
silently fall back to Phi-3.

From the repository root, the model can be downloaded with:

```bash
python - << 'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

local_directory = Path("models/llama3_8b")
local_directory.mkdir(parents=True, exist_ok=True)

model_path = hf_hub_download(
    repo_id="bartowski/Meta-Llama-3-8B-Instruct-GGUF",
    filename="Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
    local_dir=local_directory,
)

print("Downloaded model to:", model_path)
PY
```

Optional fallback support for Phi-3 exists in `MODEL_REGISTRY`, but the
notebook configuration defaults to `llama3_8b_q4_k_m`.

## How to run

From the repository root:

```bash
jupyter lab paper-empirical-explorations/01_paper_empirical_experiments_llama8b.ipynb
```

Then execute the notebook cell by cell. Run Section 5, the smoke test, before
running the empirical experiments. If exact token recovery fails, stop and fix
the implementation or environment mismatch before interpreting later results.

## Runtime notes

Defaults are CPU-friendly:

- `n_gpu_layers = 0`
- `n_ctx = 4096`
- `logits_all = True`
- small payload/key samples
- deterministic random seeds

Full-vocabulary rank computation is expensive because every token rank is
computed from the full logits vector. The cache in `results/cache/` stores
expensive `F_k(r)` calls for repeated finite-key experiments.

To increase sample sizes, edit the `CONFIG` dictionary near the top of the
notebook. Set `quick_mode=False` or increase individual fields such as
`num_payload_key_pairs`, `num_collision_payloads`, and
`num_commutativity_pairs`.

## Output locations

Each experiment saves:

- CSV tables to `results/tables/`
- raw JSON or JSONL to `results/raw/`
- PNG plots to `results/figures/`
- paper-ready Markdown summaries to `results/text/empirical_summary.md`

The notebook also writes:

- `results/tables/figure_manifest.csv`
- `results/tables/table_manifest.csv`
- `results/tables/paper_verification_checklist.csv`
- `results/raw/run_config.json`

## Experiment summary

Experiment 1 checks exact token-id round trips for `D_k(E_k(x)) = x` and the
reverse direction on encoded stegotexts.

Experiment 2 compares CARTS rank/loss statistics with ordinary greedy
generation under the same key context. The detector, when enabled, is only a
simple empirical baseline under this comparison distribution.

Experiment 3 enumerates a finite admissible key set and measures candidate
fiber sizes for `F_k(r)`.

Experiment 4 tests whether discovered finite-key collisions persist on new
payload rank vectors and measures candidate-set shrinkage across transcripts.

Experiment 5 composes key-induced rank maps and measures sampled
non-commutativity via exact equality and normalized log-rank distance.

Experiment 6 applies length-preserving token perturbations to stegotexts and
measures decoded errors.

## Security caveat

Use careful wording when reporting results:

- "under this model and tokenizer"
- "under this finite key set"
- "under this comparison distribution"
- "no collision was found in the tested sample"
- "this is not a proof of indistinguishability"

The experiments are designed to support empirical claims in a paper, not to
establish formal steganographic security.
