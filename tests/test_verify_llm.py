"""The opt-in live verification helper must not expose raw provider output."""

import asyncio
from unittest.mock import patch

import pytest
from pydantic import ValidationError

from scripts import verify_llm

pytestmark = pytest.mark.no_db


async def test_live_verification_success_omits_generated_text(capsys):
    result = {
        "entry_id": verify_llm.SAMPLE_ENTRY_ID,
        "sentiment": "positive",
        "summary": "sensitive-output-canary",
        "topics": ["sensitive-output-canary"],
    }
    with (
        patch.object(verify_llm, "get_settings"),
        patch.object(verify_llm, "analyze_journal_entry", return_value=result) as analyze,
    ):
        assert await verify_llm.main() == 0

    captured = capsys.readouterr()
    assert "Live analysis passed" in captured.out
    assert "sensitive-output-canary" not in captured.out + captured.err
    analyze.assert_awaited_once_with(verify_llm.SAMPLE_ENTRY_ID, verify_llm.SAMPLE_ENTRY_TEXT)


async def test_invalid_settings_do_not_print_input_values(capsys):
    error = ValidationError.from_exception_data(
        "Settings", [{"type": "missing", "loc": ("openai_model",), "input": "secret-canary"}]
    )
    with (
        patch.object(verify_llm, "get_settings", side_effect=error),
        patch.object(verify_llm, "analyze_journal_entry") as analyze,
    ):
        assert await verify_llm.main() == 1

    captured = capsys.readouterr()
    assert "settings are invalid" in captured.err
    assert "secret-canary" not in captured.out + captured.err
    analyze.assert_not_called()


async def test_provider_error_does_not_print_exception(capsys):
    with (
        patch.object(verify_llm, "get_settings"),
        patch.object(
            verify_llm, "analyze_journal_entry", side_effect=RuntimeError("secret-canary")
        ),
    ):
        assert await verify_llm.main() == 2

    captured = capsys.readouterr()
    assert "secret-canary" not in captured.out + captured.err


async def test_mismatched_entry_id_fails(capsys):
    result = {
        "entry_id": "wrong-id",
        "sentiment": "positive",
        "summary": "Progress.",
        "topics": ["AWS"],
    }
    with (
        patch.object(verify_llm, "get_settings"),
        patch.object(verify_llm, "analyze_journal_entry", return_value=result),
    ):
        assert await verify_llm.main() == 2

    assert "Live analysis passed" not in capsys.readouterr().out


async def test_live_verification_enforces_deadline(monkeypatch, capsys):
    async def slow_analysis(*args):
        await asyncio.Event().wait()

    monkeypatch.setattr(verify_llm, "ANALYSIS_TIMEOUT_SECONDS", 0.01)
    with (
        patch.object(verify_llm, "get_settings"),
        patch.object(verify_llm, "analyze_journal_entry", side_effect=slow_analysis) as analyze,
    ):
        assert await verify_llm.main() == 3

    assert "timed out" in capsys.readouterr().err
    analyze.assert_awaited_once()
