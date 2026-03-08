"""
Unified LLM judge backend.
Supports 'ollama' (local, default) and 'claude' (via Claude CLI subprocess).
"""
import os
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

import ollama


@dataclass(frozen=True)
class JudgeConfig:
    """Immutable judge backend configuration."""
    backend: str   # "ollama" or "claude"
    model: str     # e.g. "llama3.1:8b" or "sonnet"

    @classmethod
    def parse(cls, spec: str) -> "JudgeConfig":
        """Parse 'backend:model' string.

        Examples:
            'claude:sonnet'       → JudgeConfig("claude", "sonnet")
            'ollama:llama3.1:8b'  → JudgeConfig("ollama", "llama3.1:8b")
            'llama3.1:8b'         → JudgeConfig("ollama", "llama3.1:8b")
        """
        if spec.startswith("claude:"):
            return cls(backend="claude", model=spec[len("claude:"):])
        elif spec.startswith("ollama:"):
            return cls(backend="ollama", model=spec[len("ollama:"):])
        else:
            return cls(backend="ollama", model=spec)

    def label(self) -> str:
        return f"{self.backend}:{self.model}"


# ── Module-level default (set once at startup) ───────────────────────────────

_default_config = JudgeConfig(backend="ollama", model="llama3.1:8b")


def set_default(config: JudgeConfig) -> None:
    global _default_config
    _default_config = config


def get_default() -> JudgeConfig:
    return _default_config


# ── Main entry point ─────────────────────────────────────────────────────────

def judge_call(
    prompt: str,
    config: Optional[JudgeConfig] = None,
    timeout: int = 120,
) -> str:
    """Send a prompt to the judge LLM, return raw text response."""
    cfg = config or _default_config
    if cfg.backend == "claude":
        return _claude_call(prompt, cfg.model, timeout)
    return _ollama_call(prompt, cfg.model)


def _ollama_call(prompt: str, model: str) -> str:
    result = ollama.chat(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        options={"temperature": 0},
    )
    return result["message"]["content"]


def _claude_call(prompt: str, model: str, timeout: int) -> str:
    claude_path = shutil.which("claude")
    if not claude_path:
        raise RuntimeError(
            "Claude CLI not found. Install: npm install -g @anthropic-ai/claude-code"
        )
    # Clean env: unset CLAUDECODE to allow nested invocation
    env = {k: v for k, v in os.environ.items() if k != "CLAUDECODE"}
    try:
        result = subprocess.run(
            [claude_path, "-p", "--model", model, "--output-format", "text"],
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=env,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError(f"Claude CLI timed out after {timeout}s")

    if result.returncode != 0:
        raise RuntimeError(f"Claude CLI error (code {result.returncode}): {result.stderr.strip()}")
    return result.stdout.strip()


# ── Validation ────────────────────────────────────────────────────────────────

def check_backend_available(config: JudgeConfig) -> tuple:
    """Returns (ok: bool, message: str)."""
    if config.backend == "ollama":
        try:
            ollama.list()
            return True, "Ollama is running"
        except Exception as e:
            return False, f"Ollama not available: {e}"
    elif config.backend == "claude":
        if shutil.which("claude"):
            return True, "Claude CLI found"
        return False, "Claude CLI not found in PATH"
    return False, f"Unknown backend: {config.backend}"
