import asyncio
from collections.abc import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException
from openai import APITimeoutError

from api.config import Settings, get_settings
from api.models.entry import AnalysisResponse, Entry, EntryCreate, EntryUpdate
from api.repositories.postgres_repository import PostgresDB
from api.services.entry_service import EntryService
from api.services.llm_service import ANALYSIS_TIMEOUT_SECONDS, analyze_journal_entry

router = APIRouter()


async def get_entry_service(
    settings: Settings = Depends(get_settings),
) -> AsyncGenerator[EntryService]:
    async with PostgresDB(settings.database_url) as db:
        yield EntryService(db)


@router.post("/entries", status_code=201)
async def create_entry(
    entry_data: EntryCreate, entry_service: EntryService = Depends(get_entry_service)
):
    """Create a new journal entry."""
    # Create the full entry with auto-generated fields
    entry = Entry(
        work=entry_data.work, struggle=entry_data.struggle, intention=entry_data.intention
    )

    # Store the entry in the database
    created_entry = await entry_service.create_entry(entry.model_dump())

    # Return success response (FastAPI handles datetime serialization automatically)
    return {"detail": "Entry created successfully", "entry": created_entry}


# Implements GET /entries endpoint to list all journal entries
# Example response: [{"id": "123", "work": "...", "struggle": "...", "intention": "..."}]
@router.get("/entries")
async def get_all_entries(entry_service: EntryService = Depends(get_entry_service)):
    """Get all journal entries."""
    result = await entry_service.get_all_entries()
    return {"entries": result, "count": len(result)}


@router.get("/entries/{entry_id}")
async def get_entry(entry_id: str, entry_service: EntryService = Depends(get_entry_service)):
    """Get a single journal entry by ID."""
    entry = await entry_service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    return entry


@router.patch("/entries/{entry_id}")
async def update_entry(
    entry_id: str,
    entry_update: EntryUpdate,
    entry_service: EntryService = Depends(get_entry_service),
):
    """Update a journal entry."""
    result = await entry_service.update_entry(entry_id, entry_update.model_dump(exclude_unset=True))
    if not result:
        raise HTTPException(status_code=404, detail="Entry not found")

    return result


@router.delete("/entries/{entry_id}")
async def delete_entry(entry_id: str, entry_service: EntryService = Depends(get_entry_service)):
    """Delete a specific journal entry."""
    entry = await entry_service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    await entry_service.delete_entry(entry_id)
    return {"detail": "Entry deleted successfully"}


@router.delete("/entries")
async def delete_all_entries(entry_service: EntryService = Depends(get_entry_service)):
    """Delete all journal entries"""
    await entry_service.delete_all_entries()
    return {"detail": "All entries deleted"}


@router.post("/entries/{entry_id}/analyze", response_model=AnalysisResponse)
async def analyze_entry(entry_id: str, entry_service: EntryService = Depends(get_entry_service)):
    """
    Analyze a journal entry using AI.

    Returns sentiment, summary, key topics, entry_id, and created_at timestamp.
    The LLM call itself lives in api/services/llm_service.py.
    """
    entry = await entry_service.get_entry(entry_id)
    if entry is None:
        raise HTTPException(status_code=404, detail="Entry not found")

    entry_text = f"{entry['work']} {entry['struggle']} {entry['intention']}"

    try:
        async with asyncio.timeout(ANALYSIS_TIMEOUT_SECONDS):
            result = await analyze_journal_entry(entry_id, entry_text)
            return AnalysisResponse.model_validate(result)
    except TimeoutError, APITimeoutError:
        raise HTTPException(status_code=504, detail="Analysis timed out") from None
    except Exception:
        raise HTTPException(status_code=502, detail="Analysis service unavailable") from None
