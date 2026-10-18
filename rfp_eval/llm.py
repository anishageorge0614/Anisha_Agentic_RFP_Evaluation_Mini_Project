"""Evaluation Agent: prompt construction and provider-agnostic LLM access.

Supported providers (the user supplies the API key at runtime):
  * OpenAI-compatible chat-completions endpoints (OpenAI, Groq, OpenRouter, Gemini, Together, Ollama, vLLM, ...)
  * Anthropic Messages API
  * Offline mock (no key, deterministic keyword heuristic) for demos and tests
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

import requests

MAX_DOC_CHARS = 60_000

OPENAI_COMPATIBLE_PRESETS: dict[str, dict[str, str]] = {
    "OpenAI": {"base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini"},
    "Google Gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-2.0-flash",
    },
    "Groq": {"base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile"},
    "OpenRouter": {"base_url": "https://openrouter.ai/api/v1", "model": "openai/gpt-4o-mini"},
    "Ollama (local)": {"base_url": "http://localhost:11434/v1", "model": "llama3.1"},
    "Custom OpenAI-compatible": {"base_url": "", "model": ""},
}

SYSTEM_PROMPT = (
    "You are a meticulous procurement evaluator. You judge ONE supplier proposal at a time. "
    "You respond with a single valid JSON object and nothing else."
)


class LLMError(Exception):
    pass


class LLMFormatError(LLMError):
    """The model replied, but not with usable JSON (worth one retry)."""


@dataclass
class LLMConfig:
    provider: str = "Mock (offline demo)"  # a key of OPENAI_COMPATIBLE_PRESETS, "Anthropic" or "Mock (offline demo)"
    api_key: str = field(default="", repr=False)  # never logged or persisted
    model: str = ""
    base_url: str = ""
    temperature: float = 0.0
    timeout: int = 120

    @property
    def is_mock(self) -> bool:
        return self.provider.startswith("Mock")


def build_prompt(criteria: list[dict[str, Any]], supplier_name: str, document_text: str) -> str:
    """Build the user prompt dynamically from the criteria currently active in the database."""
    crit_lines = "\n".join(
        f'- criterion_id {c["criterion_id"]}: "{c["name"]}" (weight {c["weight"]:g}%, '
        f'score range 0 to {c["max_score"]:g}). Inspect: {c["description"]}'
        for c in criteria
    )
    example = {
        "supplier_name": supplier_name,
        "criteria": [
            {
                "criterion_id": criteria[0]["criterion_id"] if criteria else 1,
                "score": 7,
                "max_score": criteria[0]["max_score"] if criteria else 10,
                "justification": "one or two sentences explaining the score",
                "evidence": "short verbatim quote or precise paraphrase from the document",
            }
        ],
        "risks": ["short risk statement"],
        "overall_summary": "two sentence summary",
    }
    doc = document_text[:MAX_DOC_CHARS]
    truncated = " (document truncated)" if len(document_text) > MAX_DOC_CHARS else ""
    return f"""Evaluate the supplier proposal below against the active evaluation criteria.

RULES
1. Use ONLY evidence present in the supplier document. Do not use outside knowledge or assumptions.
2. Return exactly one result for EVERY active criterion listed below, using its criterion_id.
3. Each score must be a number between 0 and that criterion's max_score. Score missing or vague information low and say so in the justification.
4. "evidence" must quote or closely paraphrase the document; use "Not found in document" if there is none.
5. Do NOT compute weighted totals, rankings or comparisons with other suppliers.
6. Output JSON only: no markdown fences, no commentary.

ACTIVE CRITERIA
{crit_lines}

REQUIRED JSON SHAPE
{json.dumps(example, indent=2)}

SUPPLIER NAME: {supplier_name}

SUPPLIER DOCUMENT{truncated}
\"\"\"
{doc}
\"\"\"
"""


def extract_json_object(raw: str) -> dict[str, Any]:
    """Parse JSON from a model reply, tolerating markdown fences and surrounding prose."""
    if not isinstance(raw, str) or not raw.strip():
        raise LLMFormatError("Empty response from model")
    text = raw.strip()
    fenced = re.search(r"```(?:json)?\s*(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMFormatError("Model response did not contain a JSON object")
        try:
            obj = json.loads(text[start : end + 1])
        except json.JSONDecodeError as exc:
            raise LLMFormatError(f"Model returned malformed JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise LLMFormatError("Model JSON was not an object")
    return obj


def _post(url: str, headers: dict, payload: dict, timeout: int) -> requests.Response:
    try:
        return requests.post(url, headers=headers, json=payload, timeout=timeout)
    except requests.RequestException as exc:
        raise LLMError(f"Network error calling LLM: {exc}") from exc


def _call_openai_compatible(cfg: LLMConfig, prompt: str) -> str:
    base = cfg.base_url.rstrip("/")
    if not base or not cfg.model:
        raise LLMError("Base URL and model are required for this provider")
    headers = {"Content-Type": "application/json"}
    if cfg.api_key:
        headers["Authorization"] = f"Bearer {cfg.api_key}"
    payload: dict[str, Any] = {
        "model": cfg.model,
        "temperature": cfg.temperature,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        "response_format": {"type": "json_object"},
    }
    resp = _post(f"{base}/chat/completions", headers, payload, cfg.timeout)
    if resp.status_code == 400 and "response_format" in resp.text:
        payload.pop("response_format")  # provider without JSON mode: rely on the prompt
        resp = _post(f"{base}/chat/completions", headers, payload, cfg.timeout)
    if resp.status_code >= 400:
        raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
    try:
        return resp.json()["choices"][0]["message"]["content"]
    except (KeyError, IndexError, ValueError) as exc:
        raise LLMError(f"Unexpected LLM response format: {resp.text[:300]}") from exc


def _call_anthropic(cfg: LLMConfig, prompt: str) -> str:
    if not cfg.api_key:
        raise LLMError("An API key is required for Anthropic")
    base = (cfg.base_url or "https://api.anthropic.com").rstrip("/")
    headers = {"x-api-key": cfg.api_key, "anthropic-version": "2023-06-01", "content-type": "application/json"}
    payload = {
        "model": cfg.model or "claude-3-5-haiku-latest",
        "max_tokens": 4096,
        "temperature": cfg.temperature,
        "system": SYSTEM_PROMPT,
        "messages": [{"role": "user", "content": prompt}],
    }
    resp = _post(f"{base}/v1/messages", headers, payload, cfg.timeout)
    if resp.status_code >= 400:
        raise LLMError(f"LLM API error {resp.status_code}: {resp.text[:300]}")
    try:
        return "".join(b.get("text", "") for b in resp.json()["content"])
    except (KeyError, TypeError, ValueError) as exc:
        raise LLMError(f"Unexpected LLM response format: {resp.text[:300]}") from exc


def evaluate_supplier(
    cfg: LLMConfig, criteria: list[dict[str, Any]], supplier_name: str, document_text: str
) -> dict[str, Any]:
    """Evaluation Agent: returns the raw (unvalidated) JSON object produced by the model."""
    if cfg.is_mock:
        from .mock_llm import mock_evaluate

        return mock_evaluate(criteria, supplier_name, document_text)
    prompt = build_prompt(criteria, supplier_name, document_text)
    if cfg.provider == "Anthropic":
        raw = _call_anthropic(cfg, prompt)
    else:
        if cfg.provider not in OPENAI_COMPATIBLE_PRESETS:
            raise LLMError(f"Unknown provider: {cfg.provider}")
        if cfg.provider != "Ollama (local)" and not cfg.api_key:
            raise LLMError("An API key is required for this provider")
        raw = _call_openai_compatible(cfg, prompt)
    return extract_json_object(raw)
