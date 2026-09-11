"""Required local live verification for Task 4.

Runs ``analyze_journal_entry`` against the real OpenAI client configured
via environment variables and validates ``AnalysisResponse`` and the entry ID.
Prints timing and contract checks, not raw provider output or exceptions.

Usage:
    uv run python -m scripts.verify_llm

This script is not part of CI, which remains mocked and credential-free.
Learners must run it against a supported live LLM provider to complete
Task 4.
"""

from __future__ import annotations

import asyncio
import sys
from time import perf_counter

from openai import APITimeoutError
from pydantic import ValidationError

from api.config import get_settings
from api.models.entry import AnalysisResponse
from api.services.llm_service import ANALYSIS_TIMEOUT_SECONDS, analyze_journal_entry

SAMPLE_ENTRY_ID = "verify-llm-sample"
SAMPLE_ENTRY_TEXT = (
    "Studied FastAPI and wired up the PATCH endpoint. "
    "Struggled with understanding how async/await interacts with dependency "
    "injection. Tomorrow I'll practice writing PostgreSQL queries directly "
    "against the journal schema."
)


async def main() -> int:
    try:
        get_settings()
    except ValidationError:
        print(
            "ERROR: application settings are invalid. "
            "Check your .env file has DATABASE_URL, OPENAI_API_KEY, "
            "OPENAI_BASE_URL, and OPENAI_MODEL set.",
            file=sys.stderr,
        )
        return 1

    print(f"Calling analyze_journal_entry for entry_id={SAMPLE_ENTRY_ID!r}...")
    started = perf_counter()
    try:
        async with asyncio.timeout(ANALYSIS_TIMEOUT_SECONDS):
            result = await analyze_journal_entry(SAMPLE_ENTRY_ID, SAMPLE_ENTRY_TEXT)
            validated = AnalysisResponse.model_validate(result)
        if validated.entry_id != SAMPLE_ENTRY_ID:
            raise ValueError("Analysis entry ID does not match")
    except TimeoutError, APITimeoutError:
        print("ERROR: live analysis timed out. No automatic retry was attempted.", file=sys.stderr)
        return 3
    except Exception:
        print(
            "ERROR: live analysis failed or returned an invalid response. "
            "Check provider access, model support, and quota. "
            "Provider errors and response bodies are intentionally not printed.",
            file=sys.stderr,
        )
        return 2

    print(f"\nLive analysis passed in {perf_counter() - started:.2f} seconds.")
    print("Validated contract (model-generated text omitted):")
    print(f"  entry_id:  {validated.entry_id}")
    print(f"  sentiment: {validated.sentiment}")
    print(f"  summary:   {len(validated.summary)} characters")
    print(f"  topics:    {len(validated.topics)} non-empty strings")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
