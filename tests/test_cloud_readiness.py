"""Cloud-facing health, response contracts, deadlines, and safe errors."""

import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from httpx import AsyncClient, Request
from openai import APITimeoutError

from api.main import app
from api.routers import journal_router

pytestmark = pytest.mark.no_db


@pytest.fixture
def entry_service():
    service = AsyncMock()
    service.get_entry.return_value = {
        "id": "entry-1",
        "work": "Studied AWS",
        "struggle": "Private networking",
        "intention": "Deploy the API",
    }
    with patch.dict(app.dependency_overrides, {journal_router.get_entry_service: lambda: service}):
        yield service


@pytest.fixture
def analysis_result():
    return {
        "entry_id": "entry-1",
        "sentiment": "positive",
        "summary": "Made progress on AWS networking.",
        "topics": ["AWS", "networking"],
    }


async def test_health_needs_no_database_or_ai(test_client: AsyncClient):
    with (
        patch.object(journal_router, "PostgresDB") as database,
        patch.object(journal_router, "analyze_journal_entry") as analyze,
        patch("api.config.get_settings", side_effect=RuntimeError("No configuration")),
    ):
        response = await test_client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}
    assert response.headers["content-type"] == "application/json"
    assert "location" not in response.headers
    database.assert_not_called()
    analyze.assert_not_called()


async def test_analysis_returns_direct_json_without_auth(
    test_client: AsyncClient, entry_service, analysis_result
):
    with patch.object(journal_router, "analyze_journal_entry", return_value=analysis_result):
        response = await test_client.post("/entries/entry-1/analyze")

    assert response.status_code == 200
    assert response.headers["content-type"] == "application/json"
    assert "location" not in response.headers
    assert response.json()["entry_id"] == "entry-1"
    assert "created_at" in response.json()


async def test_analysis_missing_entry_does_not_call_provider(
    test_client: AsyncClient, entry_service
):
    entry_service.get_entry.return_value = None
    with patch.object(journal_router, "analyze_journal_entry") as analyze:
        response = await test_client.post("/entries/missing/analyze")

    assert response.status_code == 404
    analyze.assert_not_called()


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (RuntimeError("sensitive-provider-canary"), 502, "Analysis service unavailable"),
        (ValueError("sensitive-provider-canary"), 502, "Analysis service unavailable"),
        (TimeoutError("sensitive-provider-canary"), 504, "Analysis timed out"),
        (
            APITimeoutError(request=Request("POST", "https://example.invalid/v1/responses")),
            504,
            "Analysis timed out",
        ),
    ],
)
async def test_analysis_errors_are_sanitized(
    test_client: AsyncClient, entry_service, caplog, error, status, detail
):
    with patch.object(journal_router, "analyze_journal_entry", side_effect=error) as analyze:
        response = await test_client.post("/entries/entry-1/analyze")

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert "sensitive-provider-canary" not in response.text + caplog.text
    analyze.assert_awaited_once()


async def test_analysis_deadline_cancels_slow_provider(
    test_client: AsyncClient, entry_service, monkeypatch
):
    cancelled = asyncio.Event()

    async def slow_analysis(*args):
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    monkeypatch.setattr(journal_router, "ANALYSIS_TIMEOUT_SECONDS", 0.01)
    with patch.object(
        journal_router, "analyze_journal_entry", side_effect=slow_analysis
    ) as analyze:
        response = await test_client.post("/entries/entry-1/analyze")

    assert response.status_code == 504
    assert response.json() == {"detail": "Analysis timed out"}
    assert cancelled.is_set()
    analyze.assert_awaited_once()


@pytest.mark.parametrize(
    "invalid_fields",
    [
        {"sentiment": "excited"},
        {"summary": "   "},
        {"topics": []},
        {"topics": ["valid", "   "]},
    ],
)
async def test_invalid_analysis_is_a_safe_gateway_error(
    test_client: AsyncClient, entry_service, analysis_result, invalid_fields
):
    analysis_result.update(invalid_fields)
    with patch.object(journal_router, "analyze_journal_entry", return_value=analysis_result):
        response = await test_client.post("/entries/entry-1/analyze")

    assert response.status_code == 502
    assert response.json() == {"detail": "Analysis service unavailable"}
