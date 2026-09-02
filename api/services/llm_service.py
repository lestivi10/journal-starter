"""Task 4: Implement analyze_journal_entry using the OpenAI Responses API.

This project mandates the OpenAI Python SDK and a provider that supports the
Responses API, such as:
  - Microsoft Foundry Models
  - OpenAI proper

Set OPENAI_API_KEY, OPENAI_BASE_URL, and OPENAI_MODEL in your .env file.
Settings are loaded by ``api.config.Settings``.
"""

import json

from openai import AsyncOpenAI

from api.config import get_settings

ANALYSIS_INSTRUCTIONS = (
    "You are analyzing a learner's daily journal entry. Respond with a single "
    "JSON object with exactly these keys: "
    '"sentiment" (one of "positive", "negative", "neutral"), '
    '"summary" (a 2 sentence summary of the entry), and '
    '"topics" (a list of 2-4 key topics mentioned in the entry). '
    "Respond with JSON only, no other text."
)


def _default_client() -> AsyncOpenAI:
    """Construct the real OpenAI client from application settings.

    Called lazily from ``analyze_journal_entry`` so tests can inject a
    ``MockAsyncOpenAI`` without ever triggering this code path.
    """
    settings = get_settings()
    return AsyncOpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


async def analyze_journal_entry(
    entry_id: str,
    entry_text: str,
    client: AsyncOpenAI | None = None,
) -> dict:
    """Analyze a journal entry using the OpenAI Responses API.

    Args:
        entry_id: ID of the entry being analyzed (pass through to the result).
        entry_text: Combined work + struggle + intention text.
        client: OpenAI client. If None, a default one is constructed from
            application settings. Tests pass in a MockAsyncOpenAI here; production code
            in the router calls this with no ``client`` argument.

    Returns:
        A dict matching AnalysisResponse:
            {
                "entry_id":  str,
                "sentiment": str,   # "positive" | "negative" | "neutral"
                "summary":   str,
                "topics":    list[str],
            }

    """
    if client is None:
        client = _default_client()

    settings = get_settings()
    response = await client.responses.create(
        model=settings.openai_model,
        instructions=ANALYSIS_INSTRUCTIONS,
        input=f"Journal entry:\n{entry_text}",
    )

    parsed = json.loads(response.output_text)

    return {
        "entry_id": entry_id,
        "sentiment": parsed["sentiment"],
        "summary": parsed["summary"],
        "topics": parsed["topics"],
    }
