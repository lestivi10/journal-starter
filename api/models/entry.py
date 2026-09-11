from datetime import UTC, datetime
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, StringConstraints

JournalText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=256)]
NonEmptyAnalysisText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class AnalysisResponse(BaseModel):
    """Response model for journal entry analysis."""

    entry_id: NonEmptyAnalysisText = Field(description="ID of the analyzed entry")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="Sentiment: positive, negative, or neutral"
    )
    summary: NonEmptyAnalysisText = Field(description="2 sentence summary of the entry")
    topics: list[NonEmptyAnalysisText] = Field(
        min_length=1, description="Key topics mentioned in the entry (at least one)"
    )
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the analysis was created",
    )


class EntryCreate(BaseModel):
    """Model for creating a new journal entry (user input)."""

    work: JournalText = Field(
        description="What did you work on today?",
        json_schema_extra={"example": "Studied FastAPI and built my first API endpoints"},
    )
    struggle: JournalText = Field(
        description="What's one thing you struggled with today?",
        json_schema_extra={"example": "Understanding async/await syntax and when to use it"},
    )
    intention: JournalText = Field(
        description="What will you study/work on tomorrow?",
        json_schema_extra={"example": "Practice PostgreSQL queries and database design"},
    )


class EntryUpdate(BaseModel):
    """Model for partially updating a journal entry (user input)."""

    work: JournalText | None = Field(default=None, description="What did you work on today?")
    struggle: JournalText | None = Field(
        default=None, description="What's one thing you struggled with today?"
    )
    intention: JournalText | None = Field(
        default=None, description="What will you study/work on tomorrow?"
    )


class Entry(BaseModel):
    id: str = Field(
        default_factory=lambda: str(uuid4()), description="Unique identifier for the entry (UUID)."
    )
    work: str = Field(..., max_length=256, description="What did you work on today?")
    struggle: str = Field(
        ..., max_length=256, description="What's one thing you struggled with today?"
    )
    intention: str = Field(..., max_length=256, description="What will you study/work on tomorrow?")
    created_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the entry was created.",
    )
    updated_at: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="Timestamp when the entry was last updated.",
    )
