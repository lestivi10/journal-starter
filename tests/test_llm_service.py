"""Tests for Task 4: LLM-powered entry analysis.

Injects a MockAsyncOpenAI client into analyze_journal_entry, following
the pattern used by Azure-Samples/azure-search-openai-demo
(tests/test_mediadescriber.py). The mock captures calls and returns
a real ``openai.types.responses.Response`` object, so the student's
Responses API code path is exercised without making a network call.
"""

import json
from unittest.mock import AsyncMock, patch

import pytest
from openai.types.responses import Response

from api.models.entry import AnalysisResponse
from api.services import llm_service
from api.services.llm_service import analyze_journal_entry

pytestmark = pytest.mark.no_db


def _make_response(output_text: str) -> Response:
    return Response.model_validate(
        {
            "id": "resp_test",
            "created_at": 0,
            "model": "test-model",
            "object": "response",
            "output": [
                {
                    "id": "msg_test",
                    "content": [
                        {
                            "annotations": [],
                            "text": output_text,
                            "type": "output_text",
                        }
                    ],
                    "role": "assistant",
                    "status": "completed",
                    "type": "message",
                }
            ],
            "parallel_tool_calls": False,
            "tool_choice": "auto",
            "tools": [],
        }
    )


class MockResponses:
    def __init__(self, response: Response) -> None:
        self.response = response
        self.create_calls: list[dict] = []

    async def create(self, **kwargs) -> Response:
        self.create_calls.append(kwargs)
        return self.response


class MockAsyncOpenAI:
    def __init__(self, response: Response) -> None:
        self.responses = MockResponses(response)

    @property
    def create_calls(self) -> list[dict]:
        return self.responses.create_calls


SAMPLE_ENTRY_TEXT = (
    "Studied FastAPI today. Struggled with async/await syntax. "
    "Tomorrow I'll practice PostgreSQL queries."
)

VALID_ANALYSIS_JSON = json.dumps(
    {
        "sentiment": "positive",
        "summary": "Reflected on FastAPI study and async concepts.",
        "topics": ["FastAPI", "async"],
    }
)


async def test_analyze_entry_actually_calls_llm():
    client = MockAsyncOpenAI(_make_response(VALID_ANALYSIS_JSON))

    await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT, client=client)  # type: ignore[arg-type]

    assert len(client.create_calls) >= 1, (
        "Expected analyze_journal_entry to call client.responses.create() at least once."
    )


async def test_analyze_entry_sends_entry_text_in_prompt():
    client = MockAsyncOpenAI(_make_response(VALID_ANALYSIS_JSON))

    await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT, client=client)  # type: ignore[arg-type]

    call = client.create_calls[0]
    assert "input" in call
    assert "FastAPI" in json.dumps(call["input"])


async def test_analyze_entry_returns_valid_analysis_response():
    client = MockAsyncOpenAI(_make_response(VALID_ANALYSIS_JSON))

    result = await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT, client=client)  # type: ignore[arg-type]

    validated = AnalysisResponse.model_validate(result)
    assert validated.entry_id == "entry-1"
    assert validated.sentiment in {"positive", "negative", "neutral"}
    assert validated.summary
    assert isinstance(validated.topics, list)
    assert len(validated.topics) >= 1


def test_default_client_has_bounded_timeout_and_no_retries():
    with patch.object(llm_service, "AsyncOpenAI") as constructor:
        llm_service._default_client()

    assert constructor.call_args.kwargs["timeout"] == 15.0
    assert constructor.call_args.kwargs["max_retries"] == 0


@pytest.mark.parametrize("output_text", [VALID_ANALYSIS_JSON, "not-json"])
async def test_owned_client_is_closed_on_success_and_failure(output_text):
    client = MockAsyncOpenAI(_make_response(output_text))
    context_manager = AsyncMock()
    context_manager.__aenter__.return_value = client
    with patch.object(llm_service, "_default_client", return_value=context_manager):
        if output_text == VALID_ANALYSIS_JSON:
            await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT)
        else:
            with pytest.raises(json.JSONDecodeError):
                await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT)

    context_manager.__aexit__.assert_awaited_once()
    assert len(client.create_calls) == 1


@pytest.mark.parametrize(
    "output_text",
    [
        "",
        "not-json",
        "{}",
        '{"sentiment":"happy","summary":"Progress","topics":["AWS"]}',
        '{"sentiment":"positive","summary":"  ","topics":["AWS"]}',
        '{"sentiment":"positive","summary":"Progress","topics":[]}',
        '{"sentiment":"positive","summary":"Progress","topics":["  "]}',
    ],
)
async def test_invalid_model_output_is_rejected_without_retry(output_text):
    client = MockAsyncOpenAI(_make_response(output_text))
    with pytest.raises((ValueError, KeyError)):
        await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT, client=client)  # type: ignore[arg-type]

    assert len(client.create_calls) == 1


@pytest.mark.parametrize("status", ["incomplete", "failed", "cancelled", "in_progress"])
async def test_unfinished_response_is_rejected_even_with_valid_json(status):
    response = _make_response(VALID_ANALYSIS_JSON).model_copy(update={"status": status})
    client = MockAsyncOpenAI(response)
    with pytest.raises(ValueError, match="did not complete"):
        await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT, client=client)  # type: ignore[arg-type]


async def test_model_cannot_override_entry_id():
    payload = json.loads(VALID_ANALYSIS_JSON)
    payload["entry_id"] = "model-generated-wrong-id"
    client = MockAsyncOpenAI(_make_response(json.dumps(payload)))

    result = await analyze_journal_entry("entry-1", SAMPLE_ENTRY_TEXT, client=client)  # type: ignore[arg-type]

    assert result["entry_id"] == "entry-1"
