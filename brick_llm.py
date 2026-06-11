"""brick_llm.py — CANONICAL BRICK LLM router (Layer 0), Python twin.

This is the Python port of the canonical TypeScript router `llm.ts` in this
same directory. It is a faithful behavioral twin: same providers, same
provider resolution, same model tiers, same failover + circuit breaker, same
retry policy, and the same one-line `llm.call` structured log per call.

The TypeScript master serves Next.js apps; this twin serves the family's
Python work (ETL jobs — e.g. the GDM extractor Cloud Run job). When `llm.ts`
changes, this file must change with it. Architecture + rollout tracker:
02-brick.intel/docs/llm-architecture.md.

Providers (resolved + failed over automatically):
  - "ocp"       — Win-PC OCP proxy, subscription-backed, $0/call. Primary
                  when OCP_BASE_URL is set and LLM_PROVIDER != "anthropic".
  - "anthropic" — Anthropic SDK directly with ANTHROPIC_API_KEY. Fallback,
                  and forced when LLM_PROVIDER=anthropic.

Four things every caller gets for free here:
  1. Automatic failover — if OCP errors transiently, the call retries on
     Anthropic (when ANTHROPIC_API_KEY exists). A circuit breaker skips a
     flapping proxy for a cooldown window.
  2. Model tiers — ask for "fast" | "balanced" | "deep" instead of pinning a
     model ID in every caller. Bump the map here, once.
  3. Observability — every call emits a structured `llm.call` log line
     (provider, model, tier, latency, ok) for spend/usage rollups.
  4. Two return shapes — `chat()` returns just the text; `chat_detailed()`
     returns a `ChatResult` (text, usage, raw, provider) for callers that need
     token counts, the raw provider response, or the provider that actually
     answered (post-failover).

No module may import `openai` / `anthropic` outside this file. The SDKs are
imported lazily inside the call functions so the pure-function helpers (and
their tests) work without the SDKs installed — mirroring the way `llm.ts`
lazily constructs its clients.

Install deps: ``pip install openai anthropic`` (see requirements.txt).

Env contract (identical to llm.ts):
  OCP_BASE_URL, OCP_API_KEY, ANTHROPIC_API_KEY, LLM_PROVIDER,
  LLM_MODEL_FAST, LLM_MODEL_BALANCED, LLM_MODEL_DEEP, EXTRACTION_MODEL.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Dict, List, Literal, Optional, TypedDict, TypeVar

Provider = Literal["ocp", "anthropic"]
ModelTier = Literal["fast", "balanced", "deep"]


# ── model tiers ──────────────────────────────────────────────────────────
def _model_tiers() -> Dict[str, str]:
    """Tier -> model ID. One place to bump models family-wide. Keep IDs inside
    the OCP proxy allowlist.

    Resolved at call time (not import time) so env overrides set after import
    are still honored — matching the TS, which reads ``process.env`` lazily in
    practice via Node's live env object.
    """
    return {
        "fast": os.environ.get("LLM_MODEL_FAST", "claude-haiku-4-5"),
        "balanced": os.environ.get("LLM_MODEL_BALANCED", "claude-sonnet-4-6"),
        "deep": os.environ.get("LLM_MODEL_DEEP", "claude-sonnet-4-6"),
    }


# Snapshot kept for parity with the TS `export const MODEL_TIERS`. Prefer
# `_model_tiers()` internally so late env overrides are honored.
MODEL_TIERS: Dict[str, str] = _model_tiers()


def default_llm_model() -> str:
    """Default model when neither `model` nor `tier` is given."""
    return os.environ.get("EXTRACTION_MODEL") or _model_tiers()["balanced"]


# Snapshot for parity with the TS `export const DEFAULT_LLM_MODEL`.
DEFAULT_LLM_MODEL: str = default_llm_model()


def get_provider() -> Provider:
    """Resolve the primary provider from env, exactly as ``llm.ts`` does."""
    if os.environ.get("LLM_PROVIDER", "").lower() == "anthropic":
        return "anthropic"
    if os.environ.get("OCP_BASE_URL"):
        return "ocp"
    return "anthropic"


# ── lazily-constructed SDK clients ───────────────────────────────────────
_oai: Any = None
_anthropic: Any = None


def _get_openai() -> Any:
    """Lazily construct (and memoize) the OpenAI-SDK client pointed at OCP."""
    global _oai
    if _oai is not None:
        return _oai
    api_key = os.environ.get("OCP_API_KEY", "ocp-no-key")
    base_url = os.environ.get("OCP_BASE_URL")
    if not base_url:
        raise RuntimeError(
            "OCP_BASE_URL not set — cannot route to OCP proxy. Set "
            "LLM_PROVIDER=anthropic to fall back to direct Anthropic API."
        )
    from openai import OpenAI  # lazy import — see module docstring

    _oai = OpenAI(api_key=api_key, base_url=base_url)
    return _oai


def _get_anthropic() -> Any:
    """Lazily construct (and memoize) the Anthropic-SDK client."""
    global _anthropic
    if _anthropic is not None:
        return _anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Set OCP_BASE_URL to use the "
            "subscription proxy instead."
        )
    from anthropic import Anthropic  # lazy import — see module docstring

    _anthropic = Anthropic(api_key=api_key)
    return _anthropic


# ── data shapes ──────────────────────────────────────────────────────────
class ChatMessage(TypedDict):
    """A single conversation message. ``role`` is "user" | "assistant"."""

    role: str
    content: str


@dataclass
class ChatArgs:
    """Arguments for one chat round-trip."""

    #: System / instruction text. Cache hint applied on the Anthropic path.
    system: str
    #: Conversation messages — usually one user message; multi-turn supported.
    messages: List[ChatMessage]
    #: Explicit model ID. Wins over ``tier``.
    model: Optional[str] = None
    #: Capability tier — resolved via MODEL_TIERS when ``model`` is absent.
    tier: Optional[ModelTier] = None
    #: Defaults to 8192.
    max_tokens: Optional[int] = None
    #: Sampling temperature — passed through to the provider when set.
    temperature: Optional[float] = None


class ChatUsage(TypedDict, total=False):
    """Token counts, normalized across providers (absent when omitted)."""

    input_tokens: int
    output_tokens: int


@dataclass
class ChatResult:
    """Rich result — text plus the metadata callers occasionally need."""

    text: str
    usage: ChatUsage
    #: The unmodified provider response object.
    raw: Any
    #: The provider that actually produced this answer (post-failover).
    provider: Provider


# ── observability ────────────────────────────────────────────────────────
def _log_call(
    provider: Provider,
    model: str,
    tier: Optional[ModelTier],
    latency_ms: int,
    ok: bool,
    failed_over: Optional[bool] = None,
    err: Optional[str] = None,
) -> None:
    """Emit one structured ``llm.call`` line per call — cheap to grep / ship
    to a log drain later. Logging must never throw.
    """
    try:
        rec: Dict[str, Any] = {
            "provider": provider,
            "model": model,
            "latencyMs": latency_ms,
            "ok": ok,
        }
        # Match the TS JSON shape: optional keys appear only when set, and the
        # key order mirrors the TS record literal (provider, model, tier,
        # latencyMs, ok, failedOver, err, ts).
        if tier is not None:
            rec = {
                "provider": provider,
                "model": model,
                "tier": tier,
                "latencyMs": latency_ms,
                "ok": ok,
            }
        if failed_over is not None:
            rec["failedOver"] = failed_over
        if err is not None:
            rec["err"] = err
        rec["ts"] = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + (
            ".%03dZ" % (int(time.time() * 1000) % 1000)
        )
        sys.stdout.write(f"llm.call {json.dumps(rec)}\n")
    except Exception:  # noqa: BLE001 — logging must never throw
        pass


# ── circuit breaker — skip a flapping OCP proxy for a cooldown ────────────
BREAKER_THRESHOLD = 3
BREAKER_COOLDOWN_MS = 60_000

_ocp_failures = 0
_ocp_opened_at = 0.0


def _now_ms() -> float:
    """Current wall-clock time in milliseconds (parity with JS Date.now())."""
    return time.time() * 1000.0


def _ocp_breaker_open() -> bool:
    """True when the OCP circuit breaker is open (skip OCP).

    Half-open behavior: once the cooldown window has elapsed the failure
    counter is reset so a single probe call is allowed through.
    """
    global _ocp_failures
    if _ocp_failures < BREAKER_THRESHOLD:
        return False
    if _now_ms() - _ocp_opened_at > BREAKER_COOLDOWN_MS:
        _ocp_failures = 0  # cooldown elapsed — half-open, allow a probe
        return False
    return True


def _note_ocp_result(ok: bool) -> None:
    """Record an OCP call outcome, tripping the breaker on repeated failure."""
    global _ocp_failures, _ocp_opened_at
    if ok:
        _ocp_failures = 0
    else:
        _ocp_failures += 1
        if _ocp_failures >= BREAKER_THRESHOLD:
            _ocp_opened_at = _now_ms()


def _resolve_model(args: ChatArgs) -> Dict[str, Optional[str]]:
    """Resolve the model ID + tier for a call (explicit model wins over tier)."""
    if args.model:
        return {"model": args.model, "tier": args.tier}
    if args.tier:
        return {"model": _model_tiers()[args.tier], "tier": args.tier}
    return {"model": default_llm_model(), "tier": None}


# ── provider calls ───────────────────────────────────────────────────────
@dataclass
class _ProviderOutput:
    """One provider call's normalized output (text + usage + raw response)."""

    text: str
    usage: ChatUsage
    raw: Any


def _call_ocp(
    system: str,
    messages: List[ChatMessage],
    model: str,
    max_tokens: int,
    temperature: Optional[float],
) -> _ProviderOutput:
    """Call the OCP proxy via the OpenAI Python SDK's chat-completions API."""
    oai = _get_openai()
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [
            {"role": "system", "content": system},
            *({"role": m["role"], "content": m["content"]} for m in messages),
        ],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    r = oai.chat.completions.create(**kwargs)
    choice = r.choices[0] if r.choices else None
    text = ""
    if choice is not None and choice.message is not None:
        text = choice.message.content or ""
    usage: ChatUsage = {}
    if r.usage is not None:
        if r.usage.prompt_tokens is not None:
            usage["input_tokens"] = r.usage.prompt_tokens
        if r.usage.completion_tokens is not None:
            usage["output_tokens"] = r.usage.completion_tokens
    return _ProviderOutput(text=text, usage=usage, raw=r)


def _call_anthropic(
    system: str,
    messages: List[ChatMessage],
    model: str,
    max_tokens: int,
    temperature: Optional[float],
) -> _ProviderOutput:
    """Call Anthropic directly via the Anthropic Python SDK's messages API.

    The system block is marked ``cache_control: ephemeral`` → 5-min
    prompt-cache hits (~90% input-token discount) when the system prefix
    repeats.
    """
    a = _get_anthropic()
    kwargs: Dict[str, Any] = {
        "model": model,
        "max_tokens": max_tokens,
        "system": [
            {"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
    }
    if temperature is not None:
        kwargs["temperature"] = temperature
    r = a.messages.create(**kwargs)
    block = next((b for b in r.content if getattr(b, "type", None) == "text"), None)
    text = block.text if block is not None and getattr(block, "type", None) == "text" else ""
    usage: ChatUsage = {}
    if r.usage is not None:
        if getattr(r.usage, "input_tokens", None) is not None:
            usage["input_tokens"] = r.usage.input_tokens
        if getattr(r.usage, "output_tokens", None) is not None:
            usage["output_tokens"] = r.usage.output_tokens
    return _ProviderOutput(text=text, usage=usage, raw=r)


# ── core routing + failover ──────────────────────────────────────────────
def _run_chat(args: ChatArgs) -> ChatResult:
    """Core routing + failover. Returns the rich result. ``chat()`` and
    ``chat_detailed()`` are thin wrappers over this.

    Routing: primary provider per ``get_provider()``. If the primary is OCP
    and it errors transiently — or its circuit breaker is open — the call
    fails over to Anthropic when ANTHROPIC_API_KEY is set. Anthropic-primary
    does not fail over to OCP (it is the more reliable leg).
    """
    resolved = _resolve_model(args)
    model = resolved["model"] or default_llm_model()
    tier = resolved["tier"]  # type: ignore[assignment]
    max_tokens = args.max_tokens if args.max_tokens is not None else 8192
    primary = get_provider()
    anthropic_available = bool(os.environ.get("ANTHROPIC_API_KEY"))
    started = _now_ms()

    # OCP primary (unless its breaker is open) → fail over to Anthropic.
    if primary == "ocp" and not _ocp_breaker_open():
        try:
            out = _call_ocp(
                args.system, args.messages, model, max_tokens, args.temperature
            )
            _note_ocp_result(True)
            _log_call("ocp", model, tier, int(_now_ms() - started), ok=True)
            return ChatResult(
                text=out.text, usage=out.usage, raw=out.raw, provider="ocp"
            )
        except Exception as e:  # noqa: BLE001 — routed to failover / re-raised
            _note_ocp_result(False)
            err = str(e)
            if not anthropic_available:
                _log_call(
                    "ocp", model, tier, int(_now_ms() - started), ok=False, err=err
                )
                raise
            # fall through to Anthropic failover
            _log_call(
                "ocp", model, tier, int(_now_ms() - started), ok=False, err=err
            )

    fell_over = primary == "ocp"
    a_started = _now_ms()
    try:
        out = _call_anthropic(
            args.system, args.messages, model, max_tokens, args.temperature
        )
        _log_call(
            "anthropic",
            model,
            tier,
            int(_now_ms() - a_started),
            ok=True,
            failed_over=fell_over,
        )
        return ChatResult(
            text=out.text, usage=out.usage, raw=out.raw, provider="anthropic"
        )
    except Exception as e:  # noqa: BLE001 — re-raised after logging
        err = str(e)
        _log_call(
            "anthropic",
            model,
            tier,
            int(_now_ms() - a_started),
            ok=False,
            failed_over=fell_over,
            err=err,
        )
        raise


def chat(args: ChatArgs) -> str:
    """Single round-trip chat. Returns the model's text ("" if none). For
    token counts or the raw provider response, use ``chat_detailed()``.
    """
    return _run_chat(args).text


def chat_detailed(args: ChatArgs) -> ChatResult:
    """Like ``chat()``, but returns the rich ``ChatResult`` — text, usage,
    raw, and the provider that actually answered after failover.
    """
    return _run_chat(args)


# ── retry helpers ────────────────────────────────────────────────────────
LLM_MAX_ATTEMPTS = 3
LLM_BACKOFF_MS = [500, 1500, 4000]

# `ETIMEDOUT` is the connection-timeout code — listed explicitly because the
# literal `TIMEOUT` token does not match it as a substring. `TIMEOUT` still
# covers SDK-specific timeout codes (e.g. `REQUEST_TIMEOUT`).
_TRANSIENT_CODE_RE = ("ETIMEDOUT", "TIMEOUT", "ECONNRESET", "ECONNREFUSED", "EAI_AGAIN")


def is_transient_llm_error(e: Any) -> bool:
    """True when an error is worth retrying — 429, any 5xx, or a connection
    error code. Mirrors the TS ``isTransientLLMError``.
    """
    if e is None or not isinstance(e, BaseException):
        return False
    status = getattr(e, "status", None)
    if status is None:
        status = getattr(e, "status_code", None)
    if status == 429:
        return True
    if isinstance(status, int) and 500 <= status < 600:
        return True
    code = getattr(e, "code", None)
    if isinstance(code, str) and code:
        upper = code.upper()
        if any(token in upper for token in _TRANSIENT_CODE_RE):
            return True
    return False


def chat_with_retry(args: ChatArgs) -> str:
    """``chat()`` with transient-error retry. Returns just the text."""
    return chat_detailed_with_retry(args).text


def chat_detailed_with_retry(args: ChatArgs) -> ChatResult:
    """``chat_detailed()`` with up to 3 attempts on transient errors, backing
    off [500, 1500, 4000] ms between attempts.
    """
    last_err: Optional[BaseException] = None
    for attempt in range(LLM_MAX_ATTEMPTS):
        try:
            return _run_chat(args)
        except Exception as e:  # noqa: BLE001 — classified, then retried/raised
            last_err = e
            if not is_transient_llm_error(e) or attempt == LLM_MAX_ATTEMPTS - 1:
                raise
            delay_ms = (
                LLM_BACKOFF_MS[attempt] if attempt < len(LLM_BACKOFF_MS) else 4000
            )
            time.sleep(delay_ms / 1000.0)
    assert last_err is not None  # unreachable — loop always returns or raises
    raise last_err


_T = TypeVar("_T")


class _TimeoutError(RuntimeError):
    """Raised by ``with_timeout`` when the wrapped call exceeds its budget."""


def with_timeout(fn: Callable[[], _T], ms: int, label: str) -> _T:
    """Run ``fn`` with a wall-clock deadline of ``ms`` milliseconds.

    Python port of the TS ``withTimeout``. The TS version races a promise
    against a timer; Python's blocking-by-default model means we run ``fn``
    on a worker thread and abandon it if it overruns. The worker is a daemon
    thread so a stuck call never blocks process exit.
    """
    import threading

    result: Dict[str, Any] = {}

    def _runner() -> None:
        try:
            result["value"] = fn()
        except BaseException as e:  # noqa: BLE001 — surfaced to caller below
            result["error"] = e

    t = threading.Thread(target=_runner, daemon=True)
    t.start()
    t.join(ms / 1000.0)
    if t.is_alive():
        raise _TimeoutError(f"{label} timed out after {ms}ms")
    if "error" in result:
        raise result["error"]
    return result["value"]  # type: ignore[no-any-return]


# ── test seam ────────────────────────────────────────────────────────────
def _reset_breaker() -> None:
    """Reset circuit-breaker state. For tests only — not part of the public
    API and not present in ``llm.ts``.
    """
    global _ocp_failures, _ocp_opened_at
    _ocp_failures = 0
    _ocp_opened_at = 0.0
