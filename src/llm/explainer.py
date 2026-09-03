"""Provider-agnostic LLM explanation layer.

Converts structured decision inputs into concise product explanations.
This module never computes predictions — it only translates already-computed
decision data into human-readable text via a deterministic template or an
external LLM provider.
"""

from __future__ import annotations

import json
import os
from abc import ABC, abstractmethod
from typing import Any

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

class ExplanationInput(BaseModel):
    """Structured inputs that describe a decision to be explained."""

    retention_probability: float
    purchase_probability: float
    engagement_trend: float
    days_since_last_session: int
    recommended_action: str
    reasons: list[str]
    guardrails: list[str]
    user_id: str


class ExplanationOutput(BaseModel):
    """Normalised explanation returned to the caller."""

    summary: str
    reasoning: str
    action: str
    expected_objective: str
    model: str  # which provider generated the explanation


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class ExplanationProvider(ABC):
    """Base class every explanation provider must implement."""

    @abstractmethod
    def explain_decision(self, decision_input: ExplanationInput) -> ExplanationOutput:
        ...


# ---------------------------------------------------------------------------
# Deterministic (template) provider
# ---------------------------------------------------------------------------

class DeterministicFallbackProvider(ExplanationProvider):
    """Template-based explanation. Always available (no API key needed).

    Builds a human-readable explanation from the structured inputs using
    simple string formatting.  Useful as a fallback when no LLM provider
    is configured or when the LLM call fails.
    """

    def explain_decision(self, decision_input: ExplanationInput) -> ExplanationOutput:  # noqa: D401
        ret = decision_input.retention_probability
        pur = decision_input.purchase_probability
        trend = decision_input.engagement_trend
        days = decision_input.days_since_last_session

        # Summary -----------------------------------------------------------
        summary = (
            f"User {decision_input.user_id} has a {ret:.1%} retention probability, "
            f"a {pur:.1%} purchase probability, and an engagement trend of "
            f"{trend:+.2f}. The last session was {days} day(s) ago."
        )

        # Reasoning ---------------------------------------------------------
        parts: list[str] = []
        if ret >= 0.7:
            parts.append("Retention is strong — the user is likely to return.")
        elif ret >= 0.4:
            parts.append("Retention is moderate — targeted nudges may help.")
        else:
            parts.append("Retention is low — urgent re-engagement is needed.")

        if pur >= 0.5:
            parts.append("Purchase intent is high — focus on conversion.")
        elif pur >= 0.2:
            parts.append("Purchase intent is moderate — nurture the user.")
        else:
            parts.append("Purchase intent is low — awareness campaigns are appropriate.")

        if trend > 0:
            parts.append("Engagement is trending upward.")
        elif trend < -0.1:
            parts.append("Engagement is declining and requires attention.")
        else:
            parts.append("Engagement is relatively stable.")

        if days > 30:
            parts.append(f"Last session was {days} days ago — the user may be churning.")
        elif days > 7:
            parts.append(f"Last session was {days} days ago — a reminder could help.")
        else:
            parts.append("The user was recently active.")

        reasoning = " ".join(parts)

        # Action + expected objective --------------------------------------
        action = decision_input.recommended_action
        objective = (
            "Improve user retention and drive conversion by acting on the "
            "signals above."
        )

        return ExplanationOutput(
            summary=summary,
            reasoning=reasoning,
            action=action,
            expected_objective=objective,
            model="deterministic-fallback",
        )


# ---------------------------------------------------------------------------
# OpenAI provider
# ---------------------------------------------------------------------------

_OPENAI_PROMPT = """\
You are an analytics assistant. Given the structured decision data below,
produce a concise product explanation as a JSON object with exactly these
keys and value types:

{
  "summary":        "<string: 1-2 sentence overview of the user state>",
  "reasoning":      "<string: why the recommended action makes sense>",
  "action":         "<string: the recommended next action>",
  "expected_objective": "<string: what this action is expected to achieve>"
}

Decision data:
__DECISION_JSON__

Output ONLY valid JSON. Do not wrap it in markdown fences."""


class OpenAIExplanationProvider(ExplanationProvider):
    """Uses OpenAI with structured JSON output.

    Falls back to :class:`DeterministicFallbackProvider` on any failure
    (missing API key, rate limit, network error, malformed response, etc.).
    """

    def __init__(self, api_key: str, model: str = "gpt-4o-mini") -> None:
        self._api_key = api_key
        self._model = model
        self._fallback = DeterministicFallbackProvider()

    # ------------------------------------------------------------------

    def _call_openai(self, prompt: str) -> dict[str, Any]:
        """Call the OpenAI chat-completions API and return parsed JSON.

        Handles both the ``openai<1.0`` and ``openai>=1.0`` SDK styles by
        using the client-based interface that works with both.
        """
        try:
            from openai import OpenAI  # type: ignore[import-untyped]

            client = OpenAI(api_key=self._api_key)

            response = client.chat.completions.create(
                model=self._model,
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a precise analytics assistant. "
                            "Output only valid JSON matching the requested schema."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=512,
                response_format={"type": "json_object"},
            )

            raw = response.choices[0].message.content or "{}"
            return json.loads(raw)

        except Exception:  # noqa: BLE001 — intentional catch-all for fallback
            return {}

    # ------------------------------------------------------------------

    def explain_decision(self, decision_input: ExplanationInput) -> ExplanationOutput:
        """Explain the decision using an OpenAI model.

        If *any* error occurs during the API call or the response cannot be
        parsed, the method silently falls back to the deterministic template
        provider so the caller always receives a valid explanation.
        """
        decision_json = decision_input.model_dump_json(indent=2)
        prompt = _OPENAI_PROMPT.replace("__DECISION_JSON__", decision_json)

        data = self._call_openai(prompt)

        # Validate the response has the required keys
        required_keys = {"summary", "reasoning", "action", "expected_objective"}
        if not required_keys.issubset(data):
            return self._fallback.explain_decision(decision_input)

        return ExplanationOutput(
            summary=str(data["summary"]),
            reasoning=str(data["reasoning"]),
            action=str(data["action"]),
            expected_objective=str(data["expected_objective"]),
            model=self._model,
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_explanation_provider() -> ExplanationProvider:
    """Return the best available explanation provider.

    Returns an :class:`OpenAIExplanationProvider` when the ``OPENAI_API_KEY``
    environment variable is set, otherwise returns a
    :class:`DeterministicFallbackProvider`.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return OpenAIExplanationProvider(api_key=api_key)
    return DeterministicFallbackProvider()
