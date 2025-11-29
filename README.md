# LlmStenoExplore

A minimal reproduction of the "LLMs can hide text in other text of the same length" protocol  
using `llama-cpp-python` and Phi-3 Mini 3.8B Q4 GGUF.

The main demo notebook lives at:

- `notebooks/01_phi3_stego_demo.ipynb`

It shows how to:

- convert a secret text `e` (for example "Switzerland is expensive") into a sequence of token ranks,
- generate a stegotext `s` under a key prompt `k` (for example "Flour is cheap") using those ranks,
- recover the ranks from `(k, s)` and decode back to the original `e`.

---

## Quickstart (CPU only, recommended)

These commands give everyone a reproducible CPU-only setup.

```bash
# 1. Clone this repository
git clone <YOUR_GITHUB_URL_HERE> LlmStenoExplore
cd LlmStenoExplore

# 2. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate

# 3. Upgrade pip and install Python dependencies (pinned)
pip install --upgrade pip
pip install \
  "huggingface_hub==0.26.2" \
  "llama-cpp-python==0.3.5" \
  "numpy>=1.26" \
  "jupyterlab>=4.0" \
  "ipykernel>=6.29" \
  "tqdm>=4.66"

# 4. Register this environment as a Jupyter kernel
python -m ipykernel install --user --name llm-steno --display-name "Python (llm-steno)"

# 5. Download the Phi-3 Mini 4K Instruct Q4 GGUF model (about 2.3 GB)
python - << 'PY'
from pathlib import Path
from huggingface_hub import hf_hub_download

local_dir = Path("models/phi3")
local_dir.mkdir(parents=True, exist_ok=True)

path = hf_hub_download(
    repo_id="microsoft/Phi-3-mini-4k-instruct-gguf",
    filename="Phi-3-mini-4k-instruct-q4.gguf",
    local_dir=local_dir,
)

print("Downloaded model to:", path)
PY

# 6. Confirm the model file exists
ls -lh models/phi3
# You should see: Phi-3-mini-4k-instruct-q4.gguf (around 2.3G)

echo "Setup complete."


### Optional: Llama 3 8B Instruct (CPU, GGUF)

You can also run the demo with Meta Llama 3 8B Instruct in GGUF format.

**Requirements**

- A Hugging Face account with access to `meta-llama/Meta-Llama-3-8B-Instruct`.
- The `huggingface_hub` Python package (already installed in the steps above).

**Download**

From the repo root, with the virtual environment activated:

```bash
cd LlmStenoExplore
source .venv/bin/activate

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

# Check that the file is there
ls -lh models/llama3_8b
# Should show: Meta-Llama-3-8B-Instruct-Q4_K_M.gguf (~4.6G)
