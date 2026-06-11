"""Unit tests for brick_llm — pure functions only, no network, no SDKs.

Runs with pytest if installed, else as a plain script:

    python3 -m test_brick_llm        # from this directory
    python3 test_brick_llm.py

Covers: get_provider env resolution, model/tier resolution, the circuit
breaker state machine (including half-open after cooldown), and transient-
error classification. None of these touch the openai/anthropic SDKs.
"""

from __future__ import annotations

import os
import sys
from contextlib import contextmanager
from typing import Iterator

import brick_llm as L


@contextmanager
def env(**overrides: object) -> Iterator[None]:
    """Temporarily set/clear env vars; restore originals on exit."""
    saved = {k: os.environ.get(k) for k in overrides}
    try:
        for k, v in overrides.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = str(v)
        yield
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v


# ── get_provider ─────────────────────────────────────────────────────────
def test_provider_forced_anthropic() -> None:
    with env(LLM_PROVIDER="anthropic", OCP_BASE_URL="http://ocp.local"):
        assert L.get_provider() == "anthropic"


def test_provider_forced_anthropic_case_insensitive() -> None:
    with env(LLM_PROVIDER="AnThRoPiC", OCP_BASE_URL="http://ocp.local"):
        assert L.get_provider() == "anthropic"


def test_provider_ocp_when_base_url_set() -> None:
    with env(LLM_PROVIDER=None, OCP_BASE_URL="http://ocp.local"):
        assert L.get_provider() == "ocp"


def test_provider_anthropic_default_no_ocp() -> None:
    with env(LLM_PROVIDER=None, OCP_BASE_URL=None):
        assert L.get_provider() == "anthropic"


def test_provider_other_llm_provider_value_ignored() -> None:
    # LLM_PROVIDER set to something non-"anthropic" → falls back to OCP rule.
    with env(LLM_PROVIDER="ocp", OCP_BASE_URL="http://ocp.local"):
        assert L.get_provider() == "ocp"
    with env(LLM_PROVIDER="ocp", OCP_BASE_URL=None):
        assert L.get_provider() == "anthropic"


# ── model / tier resolution ──────────────────────────────────────────────
def test_default_model_ids() -> None:
    with env(
        LLM_MODEL_FAST=None,
        LLM_MODEL_BALANCED=None,
        LLM_MODEL_DEEP=None,
        EXTRACTION_MODEL=None,
    ):
        tiers = L._model_tiers()
        assert tiers["fast"] == "claude-haiku-4-5"
        assert tiers["balanced"] == "claude-sonnet-4-6"
        assert tiers["deep"] == "claude-sonnet-4-6"
        assert L.default_llm_model() == "claude-sonnet-4-6"


def test_tier_env_overrides() -> None:
    with env(
        LLM_MODEL_FAST="fast-x",
        LLM_MODEL_BALANCED="bal-x",
        LLM_MODEL_DEEP="deep-x",
    ):
        tiers = L._model_tiers()
        assert tiers == {"fast": "fast-x", "balanced": "bal-x", "deep": "deep-x"}


def test_default_model_extraction_override() -> None:
    with env(EXTRACTION_MODEL="extract-x"):
        assert L.default_llm_model() == "extract-x"


def test_resolve_model_explicit_wins_over_tier() -> None:
    args = L.ChatArgs(system="s", messages=[], model="pinned-id", tier="fast")
    assert L._resolve_model(args) == {"model": "pinned-id", "tier": "fast"}


def test_resolve_model_tier() -> None:
    with env(LLM_MODEL_DEEP="deep-x"):
        args = L.ChatArgs(system="s", messages=[], tier="deep")
        assert L._resolve_model(args) == {"model": "deep-x", "tier": "deep"}


def test_resolve_model_default() -> None:
    with env(EXTRACTION_MODEL="extract-x"):
        args = L.ChatArgs(system="s", messages=[])
        assert L._resolve_model(args) == {"model": "extract-x", "tier": None}


# ── circuit breaker state machine ────────────────────────────────────────
def test_breaker_closed_below_threshold() -> None:
    L._reset_breaker()
    L._note_ocp_result(False)
    L._note_ocp_result(False)
    assert L._ocp_failures == 2
    assert L._ocp_breaker_open() is False  # 2 < threshold of 3


def test_breaker_opens_at_threshold() -> None:
    L._reset_breaker()
    for _ in range(L.BREAKER_THRESHOLD):
        L._note_ocp_result(False)
    assert L._ocp_failures == L.BREAKER_THRESHOLD
    assert L._ocp_breaker_open() is True


def test_breaker_success_resets_counter() -> None:
    L._reset_breaker()
    L._note_ocp_result(False)
    L._note_ocp_result(False)
    L._note_ocp_result(True)  # one success clears the streak
    assert L._ocp_failures == 0
    assert L._ocp_breaker_open() is False


def test_breaker_half_open_after_cooldown() -> None:
    L._reset_breaker()
    for _ in range(L.BREAKER_THRESHOLD):
        L._note_ocp_result(False)
    assert L._ocp_breaker_open() is True
    # Simulate the cooldown window elapsing.
    L._ocp_opened_at = L._now_ms() - (L.BREAKER_COOLDOWN_MS + 1000)
    # Half-open: breaker reports closed AND resets the failure counter so a
    # single probe is allowed through.
    assert L._ocp_breaker_open() is False
    assert L._ocp_failures == 0


def test_breaker_stays_open_within_cooldown() -> None:
    L._reset_breaker()
    for _ in range(L.BREAKER_THRESHOLD):
        L._note_ocp_result(False)
    L._ocp_opened_at = L._now_ms() - 1000  # only 1s elapsed, cooldown is 60s
    assert L._ocp_breaker_open() is True


# ── transient-error classification ───────────────────────────────────────
class _Err(Exception):
    """Test error carrying optional status/code attributes."""

    def __init__(self, status: object = None, code: object = None) -> None:
        super().__init__("test error")
        if status is not None:
            self.status = status
        if code is not None:
            self.code = code


def test_transient_429() -> None:
    assert L.is_transient_llm_error(_Err(status=429)) is True


def test_transient_5xx() -> None:
    assert L.is_transient_llm_error(_Err(status=500)) is True
    assert L.is_transient_llm_error(_Err(status=503)) is True
    assert L.is_transient_llm_error(_Err(status=599)) is True


def test_transient_status_code_attr() -> None:
    # SDKs sometimes expose `status_code` rather than `status`.
    e = _Err()
    e.status_code = 502  # type: ignore[attr-defined]
    assert L.is_transient_llm_error(e) is True


def test_non_transient_4xx() -> None:
    assert L.is_transient_llm_error(_Err(status=400)) is False
    assert L.is_transient_llm_error(_Err(status=401)) is False
    assert L.is_transient_llm_error(_Err(status=404)) is False


def test_transient_conn_codes() -> None:
    # Codes matched by the TS regex
    # /ETIMEDOUT|TIMEOUT|ECONNRESET|ECONNREFUSED|EAI_AGAIN/i. `ETIMEDOUT` is
    # listed explicitly — Node's connection-timeout code does not contain the
    # literal substring "TIMEOUT".
    for code in (
        "ETIMEDOUT", "ECONNRESET", "ECONNREFUSED", "EAI_AGAIN", "REQUEST_TIMEOUT",
    ):
        assert L.is_transient_llm_error(_Err(code=code)) is True
    # case-insensitive
    assert L.is_transient_llm_error(_Err(code="econnreset")) is True
    assert L.is_transient_llm_error(_Err(code="etimedout")) is True


def test_non_transient_plain_error() -> None:
    assert L.is_transient_llm_error(Exception("boom")) is False
    assert L.is_transient_llm_error(None) is False
    assert L.is_transient_llm_error("not an error") is False
    assert L.is_transient_llm_error(_Err(code="EBADF")) is False


# ── backoff constants parity with llm.ts ─────────────────────────────────
def test_retry_constants() -> None:
    assert L.LLM_MAX_ATTEMPTS == 3
    assert L.LLM_BACKOFF_MS == [500, 1500, 4000]
    assert L.BREAKER_THRESHOLD == 3
    assert L.BREAKER_COOLDOWN_MS == 60_000


# ── plain-script runner (no pytest required) ─────────────────────────────
def _run_all() -> int:
    tests = [
        (name, obj)
        for name, obj in sorted(globals().items())
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in tests:
        try:
            fn()
            sys.stdout.write(f"PASS {name}\n")
        except AssertionError as e:
            failures += 1
            sys.stdout.write(f"FAIL {name}: {e}\n")
        except Exception as e:  # noqa: BLE001
            failures += 1
            sys.stdout.write(f"ERROR {name}: {type(e).__name__}: {e}\n")
    total = len(tests)
    sys.stdout.write(f"\n{total - failures}/{total} passed\n")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(_run_all())
