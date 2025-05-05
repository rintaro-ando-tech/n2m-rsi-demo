#!/usr/bin/env python3
"""
N2M‑RSI Self‑Prompt Divergence Experiment
----------------------------------------
Runs two self‑feedback loops with a local Llama‑cpp model:

1. Injective mode   (temperature = 1.0) – expected to diverge.
2. Deterministic    (temperature = 0.0) – expected to converge.

For each iteration we log:
    • ctx_len  – current context length in tokens
    • omega    – compression gain (proxy for information density)

Outputs:
    logs_injective_<timestamp>.json
    logs_deterministic_<timestamp>.json
Ready for downstream visualisation with matplotlib or any BI tool.
"""

import json
import zlib
from datetime import datetime
from pathlib import Path

from llama_cpp import Llama

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LLM_PATH = "/Users/LLENN/Desktop/ArXiv/Meta-Llama-3-8B-Instruct.Q4_0.gguf"  # absolute path to your local model
ITERATIONS = 10                           # Max self‑feedback steps
CTX_LIMIT = 3_500                          # Safety stop before 4 096‑token window
TEMPERATURE_INJECTIVE = 1.0                # Stochastic / injective
TEMPERATURE_DETERMINISTIC = 0.0            # Deterministic / non‑injective
TOP_P = 0.95                               # Typical nucleus sampling
REPEAT_PENALTY = 1.05                      # Mild anti‑repetition
TOP_K = 40                                # typical top‑k for injective; k=1 in deterministic

# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------
def omega_compress(text: str) -> int:
    """Return compression gain = raw_bytes − deflate_bytes (≥ 0)."""
    raw = text.encode()
    return max(0, len(raw) - len(zlib.compress(raw)))


def run_loop(temp: float) -> list[dict]:
    """
    Run a self‑prompt loop with the specified temperature.

    * temp > 0.0  → injective (stochastic) sampling
    * temp == 0.0 → fully deterministic (greedy) sampling
        - top_p           = 1.0      (no nucleus truncation)
        - repeat_penalty  = 1.0      (disable repetition penalty)
        - fixed seed      = 42       (reproducible)

    Returns
    -------
    list[dict]
        [{"t": <iter>, "ctx_len": <tokens in context>, "omega": <compression gain>}, …]
    """
    deterministic = temp == 0.0

    llm = Llama(
        model_path=LLM_PATH,
        n_ctx=4096,
        temperature=temp,
        top_p=1.0 if deterministic else TOP_P,
        top_k=1 if deterministic else TOP_K,
        repeat_penalty=1.0 if deterministic else REPEAT_PENALTY,
        seed=42 if deterministic else None,
    )

    context = ""
    logs: list[dict] = []

    for t in range(ITERATIONS):
        # choose a longer generation window for injective, shorter for deterministic
        max_tok = 8 if deterministic else 32

        completion = llm(
            context + "\n### Self:\n",
            max_tokens=max_tok,
            stop=[
                "###", "\n###",               # manual sentinel
                "<|eot_id|>",                 # Llama end‑of‑turn token
                "<|end_of_text|>"             # generic EOS
            ],
        )["choices"][0]["text"]

        # --- NEW: trim whitespace-only output in deterministic mode ---
        if deterministic:
            completion = completion.lstrip()          # drop leading newline
            if completion.strip() == "":
                # nothing substantive generated; treat as zero‑token step
                logs.append({"t": t, "ctx_len": effective_len, "omega": 0})
                break
        # ----------------------------------------------------------------

        context += completion
        # the header "\n### Self:\n" tokenizes to 2 tokens; exclude its
        # cumulative contribution so that ctx_len counts *only* model completions
        header_tok_len = 3 * (t + 1)     # accounts for "Ċ ### Self:" = 3 tokens
        effective_len = max(0, len(context.split()) - header_tok_len)

        logs.append(
            {
                "t": t,
                "ctx_len": effective_len,
                "omega": omega_compress(completion),
            }
        )

        # Safety break to avoid exceeding the context window
        if len(context.split()) > CTX_LIMIT:
            break

    return logs


def write_logs(label: str, data: list[dict]) -> None:
    """Write JSON logs to disk with a timestamped filename."""
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    Path(f"logs_{label}_{ts}.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2)
    )


# ---------------------------------------------------------------------------
# Main routine
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    write_logs("injective", run_loop(TEMPERATURE_INJECTIVE))
    write_logs("deterministic", run_loop(TEMPERATURE_DETERMINISTIC))
    print("Experiment completed – logs saved to current directory.")
