import logging

from fastapi import FastAPI

from api.routers.journal_router import router as journal_router

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("journal")
logger.info("Journal API starting up")

app = FastAPI(
    title="Journal API",
    description="A simple journal API for tracking daily work, struggles, and intentions",
)


@app.get("/health")
async def health():
    """Process health for the load balancer; no database or AI call is made."""
    return {"status": "ok"}


app.include_router(journal_router)
