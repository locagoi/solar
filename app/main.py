from fastapi import FastAPI
import logging
import os
from .maps import router as maps_router

log_level = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, log_level, logging.INFO),
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

app = FastAPI(title="FastAPI + Google Maps (minimal)")
app.include_router(maps_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
