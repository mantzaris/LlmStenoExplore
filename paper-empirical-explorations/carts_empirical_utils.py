"""Utilities for empirical CARTS experiments.

CARTS means Contextual Autoregressive Rank Transcoding Steganography.
This module keeps the core protocol at token-id level. Text helpers are
provided for display and notebook ergonomics, but exact correctness should be
checked with token ids.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_ROOT = Path(__file__).resolve().parent
RESULTS_DIR = EXPERIMENT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
TEXT_DIR = RESULTS_DIR / "text"
RAW_DIR = RESULTS_DIR / "raw"
CACHE_DIR = RESULTS_DIR / "cache"
SUMMARY_PATH = TEXT_DIR / "empirical_summary.md"


MODEL_REGISTRY = {
    "llama3_8b_q4_k_m": {
        "path": "models/llama3_8b/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf",
        "description": "Meta-Llama-3-8B-Instruct Q4_K_M GGUF",
    },
    "phi3_mini_q4": {
        "path": "models/phi3/Phi-3-mini-4k-instruct-q4.gguf",
        "description": "Phi-3 Mini 4K Instruct Q4 GGUF fallback",
    },
}


TOKENIZATION_CONVENTION = {
    "prefix_context": (
        "Non-empty prefixes are tokenized with add_bos=True and the first BOS "
        "token is dropped. Empty prefixes use the model BOS token as the "
        "minimal autoregressive context."
    ),
    "payload_text": (
        "Payload and stegotext display strings are tokenized as "
        "model.tokenize((' ' + text).encode('utf-8'), add_bos=True)[1:]."
    ),
    "rank_order": (
        "Ranks are 1-indexed. Tokens are sorted by decreasing logit with ties "
        "broken by increasing token id."
    ),
}


def ensure_dirs() -> None:
    """Create the result directory tree used by the notebook."""
    for path in [FIGURES_DIR, TABLES_DIR, TEXT_DIR, RAW_DIR, CACHE_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def _call_or_value(obj: Any) -> Any:
    return obj() if callable(obj) else obj


def get_bos_id(model: Any) -> Optional[int]:
    """Return the model BOS id when exposed by llama-cpp-python."""
    for name in ["token_bos", "bos_token_id"]:
        value = getattr(model, name, None)
        if value is not None:
            try:
                token_id = int(_call_or_value(value))
                if token_id >= 0:
                    return token_id
            except Exception:
                pass
    return None


def get_eos_id(model: Any) -> Optional[int]:
    """Return the model EOS id when exposed by llama-cpp-python."""
    for name in ["token_eos", "eos_token_id"]:
        value = getattr(model, name, None)
        if value is not None:
            try:
                token_id = int(_call_or_value(value))
                if token_id >= 0:
                    return token_id
            except Exception:
                pass
    return None


def get_vocab_size(model: Any) -> Optional[int]:
    """Best-effort vocabulary size lookup for llama-cpp-python models."""
    for name in ["n_vocab", "vocab_size"]:
        value = getattr(model, name, None)
        if value is not None:
            try:
                size = int(_call_or_value(value))
                if size > 0:
                    return size
            except Exception:
                pass
    inner_model = getattr(model, "_model", None)
    if inner_model is not None:
        for name in ["n_vocab", "vocab_size"]:
            value = getattr(inner_model, name, None)
            if value is not None:
                try:
                    size = int(_call_or_value(value))
                    if size > 0:
                        return size
                except Exception:
                    pass
    return None


def get_n_tokens(model: Any) -> int:
    value = getattr(model, "n_tokens", None)
    if value is None:
        raise AttributeError("Model does not expose n_tokens after eval().")
    return int(_call_or_value(value))


def model_identity(model: Any) -> str:
    """Stable-ish identity string for cache keys."""
    for name in ["carts_model_path", "model_path", "path"]:
        value = getattr(model, name, None)
        if value:
            return str(value)
    model_key = getattr(model, "carts_model_key", None)
    if model_key:
        return str(model_key)
    return repr(model)


def model_identity_hash(model: Any) -> str:
    return hashlib.sha256(model_identity(model).encode("utf-8")).hexdigest()[:16]


def load_language_model(
    model_key: str = "llama3_8b_q4_k_m",
    n_threads: Optional[int] = None,
    n_gpu_layers: int = 0,
    n_ctx: int = 4096,
    logits_all: bool = True,
) -> Any:
    """Load a GGUF model with llama-cpp-python.

    The notebook defaults to Llama 3 8B. If that file is missing, this raises a
    clear FileNotFoundError instead of silently falling back to another model.
    """
    if model_key not in MODEL_REGISTRY:
        known = ", ".join(sorted(MODEL_REGISTRY))
        raise KeyError(f"Unknown model_key {model_key!r}. Known keys: {known}")

    try:
        from llama_cpp import Llama
    except ImportError as exc:
        raise ImportError(
            "llama-cpp-python is required. Install llama-cpp-python in the "
            "active environment before running the CARTS notebook."
        ) from exc

    model_info = MODEL_REGISTRY[model_key]
    model_path = REPO_ROOT / model_info["path"]
    if not model_path.exists():
        if model_key == "llama3_8b_q4_k_m":
            raise FileNotFoundError(
                "Required Llama 3 8B model file was not found. Download the "
                "GGUF file to:\n"
                f"  {model_info['path']}\n"
                "Expected absolute path:\n"
                f"  {model_path}"
            )
        raise FileNotFoundError(
            f"Model file for {model_key!r} was not found at {model_path}"
        )

    if n_threads is None:
        n_threads = os.cpu_count() or 4

    model = Llama(
        model_path=str(model_path),
        n_ctx=n_ctx,
        n_gpu_layers=n_gpu_layers,
        n_threads=n_threads,
        n_batch=256,
        logits_all=logits_all,
        verbose=False,
    )

    for attr, value in [
        ("carts_model_key", model_key),
        ("carts_model_path", str(model_path)),
        ("carts_n_ctx", n_ctx),
        ("carts_logits_all", logits_all),
    ]:
        try:
            setattr(model, attr, value)
        except Exception:
            pass
    return model


def _tokenize(model: Any, text_bytes: bytes, add_bos: bool = True) -> List[int]:
    """Compatibility wrapper around llama-cpp-python tokenize."""
    try:
        return list(model.tokenize(text_bytes, add_bos=add_bos))
    except TypeError:
        if add_bos:
            return list(model.tokenize(text_bytes))
        token_ids = list(model.tokenize(text_bytes))
        bos_id = get_bos_id(model)
        if bos_id is not None and token_ids and token_ids[0] == bos_id:
            return token_ids[1:]
        return token_ids


def _make_prefix_ids(prefix: str, model: Any) -> List[int]:
    """Tokenize a prompt/key context using the repository convention."""
    if prefix:
        token_ids = _tokenize(model, prefix.encode("utf-8"), add_bos=True)
        bos_id = get_bos_id(model)
        if token_ids and bos_id is not None and token_ids[0] == bos_id:
            return token_ids[1:]
        if token_ids:
            return token_ids[1:]
        return []

    bos_id = get_bos_id(model)
    if bos_id is None:
        raise ValueError(
            "Empty prefix requested but the model does not expose a BOS token."
        )
    return [bos_id]


def text_to_payload_ids(text: str, model: Any) -> List[int]:
    """Tokenize payload/stegotext display text with the leading-space convention."""
    text = text.encode("utf-8", errors="ignore").decode("utf-8")
    token_ids = _tokenize(model, (" " + text).encode("utf-8"), add_bos=True)
    bos_id = get_bos_id(model)
    if token_ids and bos_id is not None and token_ids[0] == bos_id:
        return token_ids[1:]
    if token_ids:
        return token_ids[1:]
    return []


def safe_detokenize(token_ids: Sequence[int], model: Any) -> str:
    """Detokenize token ids with UTF-8 replacement for display."""
    try:
        raw = model.detokenize(list(map(int, token_ids)))
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="ignore")
        return str(raw)
    except Exception:
        return "".join(f"<tok:{int(token_id)}>" for token_id in token_ids)


def ids_to_text(token_ids: List[int], model: Any) -> str:
    return safe_detokenize(token_ids, model)


def _valid_admissible_ids(
    logits: np.ndarray, admissible_token_ids: Optional[Sequence[int]]
) -> np.ndarray:
    if admissible_token_ids is None:
        return np.arange(len(logits), dtype=np.int64)
    ids = np.array(list(admissible_token_ids), dtype=np.int64)
    ids = np.unique(ids)
    ids = ids[(ids >= 0) & (ids < len(logits))]
    if len(ids) == 0:
        raise ValueError("No admissible token ids remain after validation.")
    return ids


def sorted_token_ids_from_logits(
    logits: np.ndarray,
    admissible_token_ids: Optional[List[int]] = None,
) -> np.ndarray:
    """Return token ids sorted by decreasing logit, ties by increasing id."""
    scores = np.asarray(logits, dtype=np.float64)
    token_ids = _valid_admissible_ids(scores, admissible_token_ids)
    selected_scores = scores[token_ids]
    order = np.lexsort((token_ids, -selected_scores))
    return token_ids[order]


def rank_of_token(
    logits: np.ndarray,
    token_id: int,
    admissible_token_ids: Optional[List[int]] = None,
) -> int:
    sorted_ids = sorted_token_ids_from_logits(logits, admissible_token_ids)
    matches = np.where(sorted_ids == int(token_id))[0]
    if matches.size == 0:
        raise ValueError(f"Token id {token_id} is not admissible for this rank.")
    return int(matches[0]) + 1


def token_at_rank(
    logits: np.ndarray,
    rank: int,
    admissible_token_ids: Optional[List[int]] = None,
) -> int:
    sorted_ids = sorted_token_ids_from_logits(logits, admissible_token_ids)
    if rank < 1 or rank > len(sorted_ids):
        raise ValueError(
            f"Rank {rank} is outside the admissible vocabulary size {len(sorted_ids)}."
        )
    return int(sorted_ids[int(rank) - 1])


def _logsumexp(values: np.ndarray) -> float:
    finite = np.asarray(values, dtype=np.float64)
    maximum = float(np.max(finite))
    return maximum + float(np.log(np.sum(np.exp(finite - maximum))))


def logprob_of_token(
    logits: np.ndarray,
    token_id: int,
    admissible_token_ids: Optional[List[int]] = None,
) -> float:
    scores = np.asarray(logits, dtype=np.float64)
    token_ids = _valid_admissible_ids(scores, admissible_token_ids)
    if int(token_id) not in set(map(int, token_ids.tolist())):
        raise ValueError(f"Token id {token_id} is not admissible for logprob.")
    denominator = _logsumexp(scores[token_ids])
    return float(scores[int(token_id)] - denominator)


def _last_logits(model: Any) -> np.ndarray:
    n_tokens = get_n_tokens(model)
    if n_tokens <= 0:
        raise RuntimeError("Model has no evaluated tokens; cannot read logits.")
    return np.asarray(model.scores[n_tokens - 1], dtype=np.float64)


def _ensure_context_ids(context_ids: Sequence[int], model: Any) -> List[int]:
    ids = list(map(int, context_ids))
    if ids:
        return ids
    bos_id = get_bos_id(model)
    if bos_id is None:
        raise ValueError("No context ids were provided and no BOS token is available.")
    return [bos_id]


def rank_trace_from_token_ids(
    model: Any,
    context_ids: List[int],
    sequence_ids: List[int],
    admissible_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute R_c(sequence_ids) and likelihood metadata at token-id level."""
    context_ids = _ensure_context_ids(context_ids, model)
    sequence_ids = list(map(int, sequence_ids))

    model.reset()
    model.eval(context_ids)

    ranks: List[int] = []
    per_token_logprobs: List[float] = []
    per_token_nll: List[float] = []
    started_at = time.perf_counter()

    try:
        for token_id in sequence_ids:
            logits = _last_logits(model)
            rank = rank_of_token(logits, token_id, admissible_token_ids)
            logprob = logprob_of_token(logits, token_id, admissible_token_ids)
            ranks.append(rank)
            per_token_logprobs.append(logprob)
            per_token_nll.append(-logprob)
            model.eval([token_id])
        success = True
        error = None
    except Exception as exc:
        success = False
        error = repr(exc)

    mean_nll = normalized_nll(per_token_nll)
    return {
        "ranks": ranks,
        "token_ids": sequence_ids,
        "per_token_logprobs": per_token_logprobs,
        "per_token_nll": per_token_nll,
        "normalized_nll": mean_nll,
        "context_ids": context_ids,
        "first_context_ids": context_ids[:8],
        "success": success,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def generate_token_ids_from_ranks(
    model: Any,
    context_ids: List[int],
    ranks: List[int],
    admissible_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Compute G_c(ranks) and likelihood metadata at token-id level."""
    context_ids = _ensure_context_ids(context_ids, model)
    ranks = list(map(int, ranks))

    model.reset()
    model.eval(context_ids)

    generated_ids: List[int] = []
    per_token_logprobs: List[float] = []
    per_token_nll: List[float] = []
    started_at = time.perf_counter()

    try:
        for rank in ranks:
            logits = _last_logits(model)
            token_id = token_at_rank(logits, rank, admissible_token_ids)
            logprob = logprob_of_token(logits, token_id, admissible_token_ids)
            generated_ids.append(token_id)
            per_token_logprobs.append(logprob)
            per_token_nll.append(-logprob)
            model.eval([token_id])
        success = True
        error = None
    except Exception as exc:
        success = False
        error = repr(exc)

    return {
        "generated_ids": generated_ids,
        "ranks": ranks,
        "per_token_logprobs": per_token_logprobs,
        "per_token_nll": per_token_nll,
        "normalized_nll": normalized_nll(per_token_nll),
        "context_ids": context_ids,
        "generated_text": safe_detokenize(generated_ids, model),
        "success": success,
        "error": error,
        "elapsed_seconds": time.perf_counter() - started_at,
    }


def R_empty_text(text: str, model: Any, secret_prefix: str = "") -> Dict[str, Any]:
    payload_ids = text_to_payload_ids(text, model)
    context_ids = _make_prefix_ids(secret_prefix, model)
    result = rank_trace_from_token_ids(model, context_ids, payload_ids)
    result["text"] = text
    return result


def G_context_text(
    ranks: List[int],
    context: str,
    model: Any,
    secret_prefix: str = "",
) -> Dict[str, Any]:
    context_text = context if context else secret_prefix
    context_ids = _make_prefix_ids(context_text, model)
    return generate_token_ids_from_ranks(model, context_ids, ranks)


def E_key_token_ids(
    model: Any,
    payload_ids: List[int],
    key_text: str,
    secret_prefix: str = "",
    admissible_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Encode payload token ids into stegotext token ids."""
    payload_context_ids = _make_prefix_ids(secret_prefix, model)
    key_context_ids = _make_prefix_ids(key_text, model)
    rank_result = rank_trace_from_token_ids(
        model,
        payload_context_ids,
        list(map(int, payload_ids)),
        admissible_token_ids=admissible_token_ids,
    )
    generated = generate_token_ids_from_ranks(
        model,
        key_context_ids,
        rank_result["ranks"],
        admissible_token_ids=admissible_token_ids,
    )
    return {
        "payload_ids": list(map(int, payload_ids)),
        "key_text": key_text,
        "secret_prefix": secret_prefix,
        "ranks": rank_result["ranks"],
        "payload_rank_trace": rank_result,
        "stegotext_ids": generated["generated_ids"],
        "stegotext_text": generated["generated_text"],
        "stegotext_generation": generated,
        "success": bool(rank_result["success"] and generated["success"]),
    }


def D_key_token_ids(
    model: Any,
    stegotext_ids: List[int],
    key_text: str,
    secret_prefix: str = "",
    admissible_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Decode stegotext token ids into payload token ids."""
    key_context_ids = _make_prefix_ids(key_text, model)
    payload_context_ids = _make_prefix_ids(secret_prefix, model)
    stego_rank_result = rank_trace_from_token_ids(
        model,
        key_context_ids,
        list(map(int, stegotext_ids)),
        admissible_token_ids=admissible_token_ids,
    )
    decoded = generate_token_ids_from_ranks(
        model,
        payload_context_ids,
        stego_rank_result["ranks"],
        admissible_token_ids=admissible_token_ids,
    )
    return {
        "stegotext_ids": list(map(int, stegotext_ids)),
        "key_text": key_text,
        "secret_prefix": secret_prefix,
        "recovered_ranks": stego_rank_result["ranks"],
        "stegotext_rank_trace": stego_rank_result,
        "decoded_ids": decoded["generated_ids"],
        "decoded_text": decoded["generated_text"],
        "decoded_generation": decoded,
        "success": bool(stego_rank_result["success"] and decoded["success"]),
    }


def encode_text(
    payload_text: str,
    key_text: str,
    model: Any,
    secret_prefix: str = "",
) -> Dict[str, Any]:
    payload_ids = text_to_payload_ids(payload_text, model)
    result = E_key_token_ids(model, payload_ids, key_text, secret_prefix)
    result["payload_text"] = payload_text
    return result


def decode_text(
    stegotext_text: str,
    key_text: str,
    model: Any,
    secret_prefix: str = "",
) -> Dict[str, Any]:
    stego_ids = text_to_payload_ids(stegotext_text, model)
    result = D_key_token_ids(model, stego_ids, key_text, secret_prefix)
    result["stegotext_text_input"] = stegotext_text
    return result


def _cache_key_for_f(
    model: Any,
    ranks: Sequence[int],
    key_text: str,
    secret_prefix: str,
    admissible_token_ids: Optional[Sequence[int]],
) -> str:
    admissible_digest = "all"
    if admissible_token_ids is not None:
        admissible_payload = list(map(int, admissible_token_ids))
        admissible_digest = hashlib.sha256(
            json.dumps(admissible_payload, sort_keys=True).encode("utf-8")
        ).hexdigest()
    payload = {
        "model_identity_hash": model_identity_hash(model),
        "model_identity": model_identity(model),
        "key_text": key_text,
        "ranks": list(map(int, ranks)),
        "secret_prefix": secret_prefix,
        "admissible_token_ids": admissible_digest,
        "rank_order": TOKENIZATION_CONVENTION["rank_order"],
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()


def F_key_on_ranks(
    ranks: List[int],
    key_text: str,
    model: Any,
    secret_prefix: str = "",
    admissible_token_ids: Optional[List[int]] = None,
    use_cache: bool = True,
) -> List[int]:
    """Compute F_k(r) = R_empty(G_k(r)) with optional disk caching."""
    ensure_dirs()
    cache_digest = _cache_key_for_f(
        model,
        ranks,
        key_text,
        secret_prefix,
        admissible_token_ids,
    )
    cache_path = CACHE_DIR / f"f_key_{cache_digest}.json"
    if use_cache and cache_path.exists():
        cached = read_json(cache_path)
        return list(map(int, cached["output_ranks"]))

    key_context_ids = _make_prefix_ids(key_text, model)
    generated = generate_token_ids_from_ranks(
        model,
        key_context_ids,
        list(map(int, ranks)),
        admissible_token_ids=admissible_token_ids,
    )
    empty_context_ids = _make_prefix_ids(secret_prefix, model)
    transformed = rank_trace_from_token_ids(
        model,
        empty_context_ids,
        generated["generated_ids"],
        admissible_token_ids=admissible_token_ids,
    )
    output_ranks = list(map(int, transformed["ranks"]))

    if use_cache:
        write_json(
            cache_path,
            {
                "key_text": key_text,
                "input_ranks": list(map(int, ranks)),
                "output_ranks": output_ranks,
                "secret_prefix": secret_prefix,
                "model_identity": model_identity(model),
                "generated_ids": generated["generated_ids"],
            },
        )
    return output_ranks


def ordinary_greedy_generation(
    model: Any,
    key_text: str,
    length_n: int,
    secret_prefix: str = "",
    admissible_token_ids: Optional[List[int]] = None,
) -> Dict[str, Any]:
    """Generate ordinary greedy model text under key_text for exactly length_n tokens."""
    context_text = key_text if key_text else secret_prefix
    context_ids = _make_prefix_ids(context_text, model)
    return generate_token_ids_from_ranks(
        model,
        context_ids,
        [1] * int(length_n),
        admissible_token_ids=admissible_token_ids,
    )


def ordinary_sample_generation(
    model: Any,
    key_text: str,
    length_n: int,
    secret_prefix: str = "",
    admissible_token_ids: Optional[List[int]] = None,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 123,
) -> Dict[str, Any]:
    """Optional deterministic-seed nucleus sampling comparison generator."""
    rng = np.random.default_rng(seed)
    context_text = key_text if key_text else secret_prefix
    context_ids = _make_prefix_ids(context_text, model)
    context_ids = _ensure_context_ids(context_ids, model)

    model.reset()
    model.eval(context_ids)

    generated_ids: List[int] = []
    ranks: List[int] = []
    per_token_logprobs: List[float] = []
    per_token_nll: List[float] = []

    for _ in range(int(length_n)):
        logits = _last_logits(model)
        sorted_ids = sorted_token_ids_from_logits(logits, admissible_token_ids)
        filtered_logits = logits[sorted_ids] / max(float(temperature), 1e-8)
        log_denom = _logsumexp(filtered_logits)
        probs = np.exp(filtered_logits - log_denom)
        cumulative = np.cumsum(probs)
        cutoff = int(np.searchsorted(cumulative, top_p, side="left")) + 1
        cutoff = max(1, min(cutoff, len(sorted_ids)))
        nucleus_ids = sorted_ids[:cutoff]
        nucleus_probs = probs[:cutoff]
        nucleus_probs = nucleus_probs / np.sum(nucleus_probs)
        token_id = int(rng.choice(nucleus_ids, p=nucleus_probs))
        rank = rank_of_token(logits, token_id, admissible_token_ids)
        logprob = logprob_of_token(logits, token_id, admissible_token_ids)
        generated_ids.append(token_id)
        ranks.append(rank)
        per_token_logprobs.append(logprob)
        per_token_nll.append(-logprob)
        model.eval([token_id])

    return {
        "generated_ids": generated_ids,
        "ranks": ranks,
        "per_token_logprobs": per_token_logprobs,
        "per_token_nll": per_token_nll,
        "normalized_nll": normalized_nll(per_token_nll),
        "context_ids": context_ids,
        "generated_text": safe_detokenize(generated_ids, model),
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "success": True,
        "error": None,
    }


def normalized_rank_hamming(u: Sequence[int], v: Sequence[int]) -> float:
    maximum = max(len(u), len(v), 1)
    mismatches = 0
    for idx in range(maximum):
        a = u[idx] if idx < len(u) else None
        b = v[idx] if idx < len(v) else None
        if a != b:
            mismatches += 1
    return mismatches / maximum


def normalized_log_rank_distance(
    u: Sequence[int],
    v: Sequence[int],
    vocab_size: int,
) -> float:
    maximum = max(len(u), len(v), 1)
    denom = max(math.log(max(int(vocab_size), 2)), 1e-12)
    total = 0.0
    for idx in range(maximum):
        a = int(u[idx]) if idx < len(u) else int(vocab_size)
        b = int(v[idx]) if idx < len(v) else int(vocab_size)
        total += abs(math.log(max(a, 1)) - math.log(max(b, 1))) / denom
    return total / maximum


def token_edit_distance(a_ids: Sequence[int], b_ids: Sequence[int]) -> int:
    """Levenshtein edit distance over token ids using a small DP implementation."""
    a = list(a_ids)
    b = list(b_ids)
    if len(a) < len(b):
        a, b = b, a
    previous = list(range(len(b) + 1))
    for i, token_a in enumerate(a, start=1):
        current = [i]
        for j, token_b in enumerate(b, start=1):
            insert_cost = current[j - 1] + 1
            delete_cost = previous[j] + 1
            substitute_cost = previous[j - 1] + (0 if token_a == token_b else 1)
            current.append(min(insert_cost, delete_cost, substitute_cost))
        previous = current
    return previous[-1]


def normalized_token_edit_distance(a_ids: Sequence[int], b_ids: Sequence[int]) -> float:
    return token_edit_distance(a_ids, b_ids) / max(len(a_ids), len(b_ids), 1)


def mean_log_rank(ranks: Sequence[int]) -> float:
    if not ranks:
        return float("nan")
    return float(np.mean([math.log(max(int(rank), 1)) for rank in ranks]))


def max_log_rank(ranks: Sequence[int]) -> float:
    if not ranks:
        return float("nan")
    return float(max(math.log(max(int(rank), 1)) for rank in ranks))


def tail_rank_fraction(ranks: Sequence[int], B: int) -> float:
    if not ranks:
        return float("nan")
    threshold = int(B)
    return float(sum(1 for rank in ranks if int(rank) > threshold) / len(ranks))


def normalized_nll(per_token_nll: Sequence[float]) -> float:
    if not per_token_nll:
        return float("nan")
    return float(np.mean(np.asarray(per_token_nll, dtype=np.float64)))


def first_mismatch_position(a: Sequence[Any], b: Sequence[Any]) -> Optional[int]:
    maximum = max(len(a), len(b))
    for idx in range(maximum):
        left = a[idx] if idx < len(a) else None
        right = b[idx] if idx < len(b) else None
        if left != right:
            return idx
    return None


def suffix_corruption_fraction(a: Sequence[Any], b: Sequence[Any]) -> float:
    mismatch = first_mismatch_position(a, b)
    if mismatch is None:
        return 0.0
    maximum = max(len(a), len(b), 1)
    suffix_len = maximum - mismatch
    if suffix_len <= 0:
        return 0.0
    suffix_mismatches = 0
    for idx in range(mismatch, maximum):
        left = a[idx] if idx < len(a) else None
        right = b[idx] if idx < len(b) else None
        if left != right:
            suffix_mismatches += 1
    return suffix_mismatches / suffix_len


def exact_match(a: Sequence[Any], b: Sequence[Any]) -> bool:
    return list(a) == list(b)


def simple_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    """Threshold-free AUC via pairwise Mann-Whitney comparison."""
    positives = [score for label, score in zip(labels, scores) if int(label) == 1]
    negatives = [score for label, score in zip(labels, scores) if int(label) == 0]
    if not positives or not negatives:
        return float("nan")
    wins = 0.0
    total = 0
    for pos in positives:
        for neg in negatives:
            if pos > neg:
                wins += 1.0
            elif pos == neg:
                wins += 0.5
            total += 1
    return wins / total


try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score as _sklearn_roc_auc_score
    from sklearn.model_selection import train_test_split

    SKLEARN_AVAILABLE = True
except Exception:
    LogisticRegression = None
    _sklearn_roc_auc_score = None
    train_test_split = None
    SKLEARN_AVAILABLE = False


def roc_auc_score(labels: Sequence[int], scores: Sequence[float]) -> float:
    if SKLEARN_AVAILABLE and _sklearn_roc_auc_score is not None:
        return float(_sklearn_roc_auc_score(labels, scores))
    return simple_auc_score(labels, scores)


def train_simple_logistic_regression(
    feature_rows: Sequence[Sequence[float]],
    labels: Sequence[int],
    random_seed: int = 123,
    test_size: float = 0.35,
) -> Dict[str, Any]:
    """Train a small logistic detector if sklearn is available."""
    if not SKLEARN_AVAILABLE:
        return {
            "available": False,
            "note": "sklearn is unavailable; logistic regression detector skipped.",
            "auc": float("nan"),
            "scores": [],
            "labels": list(map(int, labels)),
        }
    X = np.asarray(feature_rows, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if len(np.unique(y)) < 2 or len(y) < 4:
        return {
            "available": False,
            "note": "not enough labeled examples for logistic regression.",
            "auc": float("nan"),
            "scores": [],
            "labels": y.tolist(),
        }
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
            stratify=y,
        )
    except ValueError:
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_seed,
        )
    clf = LogisticRegression(max_iter=1000, random_state=random_seed)
    clf.fit(X_train, y_train)
    scores = clf.predict_proba(X_test)[:, 1]
    auc = float(_sklearn_roc_auc_score(y_test, scores))
    return {
        "available": True,
        "note": "sklearn logistic regression detector trained on a small split.",
        "auc": auc,
        "scores": scores.tolist(),
        "labels": y_test.tolist(),
        "coef": clf.coef_.tolist(),
        "intercept": clf.intercept_.tolist(),
    }


def _prepare_figure(filename: str) -> Path:
    ensure_dirs()
    return FIGURES_DIR / filename


def _save_figure(fig: Any, filename: str) -> Path:
    output_path = _prepare_figure(filename)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    return output_path


def histogram(
    values: Sequence[float],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str = "Count",
    bins: int = 20,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.hist(list(values), bins=bins)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    _save_figure(fig, filename)
    return fig


def empirical_cdf(
    series_by_label: Dict[str, Sequence[float]],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str = "Cumulative fraction",
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    for label, values in series_by_label.items():
        arr = np.sort(np.asarray(list(values), dtype=np.float64))
        if len(arr) == 0:
            continue
        y = np.arange(1, len(arr) + 1) / len(arr)
        ax.plot(arr, y, marker=".", linestyle="-", label=label)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, filename)
    return fig


def boxplot(
    data_by_label: Dict[str, Sequence[float]],
    filename: str,
    title: str,
    ylabel: str,
    xlabel: str = "",
) -> Any:
    import matplotlib.pyplot as plt

    labels = list(data_by_label.keys())
    values = [list(data_by_label[label]) for label in labels]
    fig, ax = plt.subplots()
    ax.boxplot(values, labels=labels, showmeans=True)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, filename)
    return fig


def scatter_plot(
    x: Sequence[float],
    y: Sequence[float],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    labels: Optional[Sequence[str]] = None,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    if labels is None:
        ax.scatter(x, y, alpha=0.75)
    else:
        unique_labels = list(dict.fromkeys(labels))
        for label in unique_labels:
            xs = [xv for xv, lv in zip(x, labels) if lv == label]
            ys = [yv for yv, lv in zip(y, labels) if lv == label]
            ax.scatter(xs, ys, alpha=0.75, label=label)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, filename)
    return fig


def heatmap(
    matrix: Sequence[Sequence[float]],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    xticklabels: Optional[Sequence[str]] = None,
    yticklabels: Optional[Sequence[str]] = None,
    colorbar_label: str = "",
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Any:
    import matplotlib.pyplot as plt

    arr = np.asarray(matrix, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(8, 5))
    image = ax.imshow(arr, aspect="auto", interpolation="nearest", vmin=vmin, vmax=vmax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if xticklabels is not None:
        ax.set_xticks(range(len(xticklabels)))
        ax.set_xticklabels(list(xticklabels), rotation=45, ha="right")
    if yticklabels is not None:
        ax.set_yticks(range(len(yticklabels)))
        ax.set_yticklabels(list(yticklabels))
    fig.colorbar(image, ax=ax, label=colorbar_label)
    fig.tight_layout()
    _save_figure(fig, filename)
    return fig


def line_plot(
    x: Sequence[float],
    y: Sequence[float],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
    label: Optional[str] = None,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.plot(x, y, marker="o", label=label)
    if label:
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    _save_figure(fig, filename)
    return fig


def bar_plot(
    labels: Sequence[str],
    values: Sequence[float],
    filename: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> Any:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots()
    ax.bar(list(labels), list(values))
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    _save_figure(fig, filename)
    return fig


def roc_curve_plot(
    labels: Sequence[int],
    scores: Sequence[float],
    filename: str,
    title: str,
) -> Any:
    import matplotlib.pyplot as plt

    pairs = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    positives = sum(1 for label in labels if int(label) == 1)
    negatives = sum(1 for label in labels if int(label) == 0)
    tp = 0
    fp = 0
    fpr = [0.0]
    tpr = [0.0]
    for _, label in pairs:
        if int(label) == 1:
            tp += 1
        else:
            fp += 1
        tpr.append(tp / max(positives, 1))
        fpr.append(fp / max(negatives, 1))
    fpr.append(1.0)
    tpr.append(1.0)
    auc = roc_auc_score(labels, scores)

    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, marker=".", label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title(title)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.grid(True, alpha=0.3)
    ax.legend()
    _save_figure(fig, filename)
    return fig


def reset_summary(title: str = "CARTS Empirical Summary") -> None:
    ensure_dirs()
    SUMMARY_PATH.write_text(f"# {title}\n\n", encoding="utf-8")


def append_summary(section_title: str, markdown_text: str) -> None:
    ensure_dirs()
    with SUMMARY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(f"\n## {section_title}\n\n")
        handle.write(markdown_text.rstrip() + "\n")


def _json_default(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        value = float(obj)
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return str(obj)


def write_json(path: Path, obj: Any) -> Path:
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(obj, handle, indent=2, sort_keys=True, default=_json_default)
    return path


def read_json(path: Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Path:
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True, default=_json_default) + "\n")
    return path


def _rows_and_fieldnames(rows: Sequence[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[str]]:
    materialized = list(rows)
    fieldnames: List[str] = []
    for row in materialized:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    return materialized, fieldnames


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> Path:
    ensure_dirs()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    materialized, fieldnames = _rows_and_fieldnames(rows)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in materialized:
            clean_row = {
                key: json.dumps(value, default=_json_default)
                if isinstance(value, (list, dict, tuple))
                else value
                for key, value in row.items()
            }
            writer.writerow(clean_row)
    return path


def write_table(path: Path, dataframe_or_rows: Any) -> Path:
    if hasattr(dataframe_or_rows, "to_csv"):
        ensure_dirs()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        dataframe_or_rows.to_csv(path, index=False)
        return path
    return write_csv(path, list(dataframe_or_rows))


def environment_info() -> Dict[str, Any]:
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "sklearn_available": SKLEARN_AVAILABLE,
    }
    try:
        import llama_cpp

        info["llama_cpp"] = getattr(llama_cpp, "__version__", "unknown")
    except Exception:
        info["llama_cpp"] = "not importable"
    try:
        import matplotlib

        info["matplotlib"] = getattr(matplotlib, "__version__", "unknown")
    except Exception:
        info["matplotlib"] = "not importable"
    return info


def save_manifest(config: Dict[str, Any], environment: Dict[str, Any]) -> Path:
    payload = {
        "config": config,
        "environment": environment,
        "paths": {
            "repo_root": str(REPO_ROOT),
            "experiment_root": str(EXPERIMENT_ROOT),
            "results_dir": str(RESULTS_DIR),
            "figures_dir": str(FIGURES_DIR),
            "tables_dir": str(TABLES_DIR),
            "text_dir": str(TEXT_DIR),
            "raw_dir": str(RAW_DIR),
            "cache_dir": str(CACHE_DIR),
        },
        "tokenization_convention": TOKENIZATION_CONVENTION,
        "created_at_unix": time.time(),
    }
    return write_json(RAW_DIR / "run_config.json", payload)


def list_saved_figures() -> List[Path]:
    ensure_dirs()
    return sorted(FIGURES_DIR.glob("*.png"))


def list_saved_tables() -> List[Path]:
    ensure_dirs()
    return sorted(TABLES_DIR.glob("*.csv"))


def deterministic_choice(items: Sequence[Any], seed: int, n: int) -> List[Any]:
    rng = random.Random(seed)
    items = list(items)
    if n >= len(items):
        return items
    return rng.sample(items, n)
