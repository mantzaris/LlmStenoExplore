# CARTS empirical explorations

This directory contains the empirical notebook and supporting utilities for a paper on CARTS: Contextual Autoregressive Rank Transcoding Steganography.

The current default model is Meta-Llama-3-8B-Instruct Q4_K_M GGUF through `llama-cpp-python`. The notebook does not silently switch to Phi-3.

## Files

- `01_paper_empirical_experiments_llama8b.ipynb`: research notebook for the six core experiments.
- `carts_empirical_utils.py`: token-id-level CARTS primitives, metrics, cache, plotting helpers, CSV/JSON writers, confidence intervals, and run manifests.
- `results/figures/`: generated PNG figures.
- `results/tables/`: generated CSV tables and manifests.
- `results/text/`: generated Markdown summaries.
- `results/raw/`: raw JSON/JSONL result files and `run_manifest.json`.
- `results/cache/`: disk cache for expensive `F_k(r)` computations.
- `results_previous_runs/<timestamp>/`: backups of previous results.

## Required model

The main model path is:

```text
models/llama3_8b/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf
```

If this file is missing, the loader raises a clear error and stops. Optional Phi-3 support exists only as a manual fallback in `MODEL_REGISTRY`.

Download from the repository root with:

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

## Run profiles

The notebook defines three profiles:

- `smoke`: small CPU-safe validation run.
- `paper_medium`: recommended paper rerun; default in the notebook.
- `paper_full`: larger profile for explicit long runs only; do not run automatically.

`paper_medium` uses 40 payloads, 120 finite keys, 80 correctness/detection pairs, 40 collision transcripts, 100 non-commutativity key pairs, and 40 robustness base cases. On CPU this can take many hours because full-vocabulary rank computations are repeated for `F_k(r)`. The cache in `results/cache/` lets interrupted or repeated runs resume many expensive map evaluations.

To run the default paper profile from the repository root:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  paper-empirical-explorations/01_paper_empirical_experiments_llama8b.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=calgacus-repl
```

To run the CPU-safe smoke profile without editing the notebook:

```bash
CARTS_RUN_PROFILE=smoke jupyter nbconvert --to notebook --execute --inplace \
  paper-empirical-explorations/01_paper_empirical_experiments_llama8b.ipynb \
  --ExecutePreprocessor.timeout=-1 \
  --ExecutePreprocessor.kernel_name=calgacus-repl
```

Use the kernel that points at this repository virtualenv. On this machine that kernel is `calgacus-repl`.

## Backups and reruns

Before overwriting results, back up the previous run:

```bash
ts=$(date +%Y%m%d_%H%M%S)
mkdir -p paper-empirical-explorations/results_previous_runs/$ts
cp -a paper-empirical-explorations/results/. paper-empirical-explorations/results_previous_runs/$ts/
```

The notebook clears `results/figures`, `results/tables`, `results/text`, and `results/raw` at the start of a run, while preserving `results/cache` by default. To force a fully cold run, clear `results/cache` manually after making a backup.

## Experiments

1. Implementation correctness checks exact token-id recovery for `D_k(E_k(x)) = x` and reverse recovery on encoded stegotexts.
2. Rank-likelihood and simple detection compares CARTS against `Q_greedy` and `Q_sampled`.
3. Key-collision fiber census enumerates finite `K_adm` and searches for candidate fibers of `F_k(r)`.
4. Collision stability runs only when collisions are found; candidate shrinkage always runs.
5. Non-commutativity compares `F_k(F_h(r))` with `F_h(F_k(r))` on sampled rank vectors.
6. Robustness applies length-preserving stegotext token perturbations and decodes the result.

## Comparison distributions

`Q_greedy` is ordinary greedy generation under the same key context and exact token length. It is useful as a deterministic baseline but is weak: rank traces are concentrated at rank 1.

`Q_sampled` is ordinary stochastic generation under the same key context with temperature 0.8, top-p 0.95, fixed seed, and exact token length. It is a stronger baseline than greedy but still not a complete cover-channel model.

Detector AUCs are distribution-specific. A high AUC means the tested CARTS outputs were separable from the chosen comparison distribution under the tested features; it is not a proof that CARTS is insecure.

## Interpreting zero collisions

If Experiment 3 reports zero collisions, use this wording:

> No collisions were found under this finite key set and sample.

Do not infer that collisions are impossible. The finite search says nothing conclusive about the unrestricted prompt space.

## Outputs

The run writes:

- `results/text/empirical_summary.md`
- `results/raw/run_manifest.json`
- `results/raw/run_config.json`
- `results/tables/figure_manifest.csv`
- `results/tables/table_manifest.csv`
- `results/tables/paper_verification_checklist.csv`
- one CSV and raw JSON file per experiment
- PNG figures for each experiment

All manifest and summary paths are relative to `paper-empirical-explorations/`.

## Syntax checks

Run these after editing:

```bash
python -m py_compile paper-empirical-explorations/carts_empirical_utils.py
python -m compileall paper-empirical-explorations
```

## Security caveat

These experiments do not prove steganographic security, insecurity, or indistinguishability. Report results with careful qualifiers such as:

- under this model and tokenizer
- under this finite key set
- under this comparison distribution
- no collision was found in the tested sample
- this is not a proof of indistinguishability
