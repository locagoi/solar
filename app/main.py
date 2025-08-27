from fastapi import FastAPI
from .maps import router as maps_router

app = FastAPI(title="FastAPI + Google Maps (minimal)")
app.include_router(maps_router)

@app.get("/health")
async def health():
    return {"status": "ok"}
