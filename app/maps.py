import os
import re
import math
import base64
import httpx
import json
import logging

from io import BytesIO
from urllib.parse import urlparse, parse_qs, unquote
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query

# ---------- OpenAI ----------
from openai import OpenAI, AsyncOpenAI

logger = logging.getLogger(__name__)

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)

def analyze_image_bytes_with_openai(image_bytes: bytes, prompt: str, model: str = "gpt-5") -> str:
    """
    Send a PNG (bytes) + prompt to OpenAI Responses API and return text.
    Uses a base64 data URL so no hosting is needed.
    """
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")

    try:
        client = _get_openai_client()
        resp = client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
        )
        # Prefer output_text if available; otherwise fall back to stringifying.
        return getattr(resp, "output_text", None) or str(resp)
    except Exception as e:
        logger.exception("OpenAI request failed: %s", e)
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

async def analyze_image_bytes_with_openai_async(image_bytes: bytes, prompt: str, model: str = "gpt-5") -> str:
    """
    Async variant that avoids blocking the event loop during OpenAI network I/O.
    """
    data_url = "data:image/png;base64," + base64.b64encode(image_bytes).decode("ascii")
    try:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
        client = AsyncOpenAI(api_key=api_key)
        resp = await client.responses.create(
            model=model,
            input=[{
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": data_url},
                ],
            }],
        )
        return getattr(resp, "output_text", None) or str(resp)
    except Exception as e:
        logger.exception("OpenAI async request failed: %s", e)
        raise HTTPException(status_code=502, detail=f"OpenAI error: {e}")

# ---------- Google Maps ----------
router = APIRouter(prefix="/maps", tags=["maps"])
GOOGLE_BASE = "https://maps.googleapis.com/maps/api"

def make_circle_path(lat: float, lng: float, radius_m: int = 200, num_points: int = 60) -> str:
    R = 6378137.0
    lat_rad = math.radians(lat)
    lng_rad = math.radians(lng)
    coords = []
    for i in range(num_points):
        angle = 2 * math.pi * i / num_points
        dlat = (radius_m / R) * math.cos(angle)
        dlng = (radius_m / (R * math.cos(lat_rad))) * math.sin(angle)
        lat_i = lat_rad + dlat
        lng_i = lng_rad + dlng
        coords.append(f"{math.degrees(lat_i)},{math.degrees(lng_i)}")
    return "|".join(coords)

def extract_coords_from_url(url: str):
    m = re.search(r'@([-+0-9.]+),([-+0-9.]+),\d+(?:\.\d+)?z', url)
    if m:
        return float(m.group(1)), float(m.group(2))
    m = re.search(r'!3d([-+0-9.]+)!4d([-+0-9.]+)', url)
    if m:
        return float(m.group(1)), float(m.group(2))
    qs = parse_qs(urlparse(url).query)
    q_vals = qs.get("q") or qs.get("query") or []
    if q_vals:
        q = unquote(q_vals[0])
        m = re.match(r'\s*([-+0-9.]+)\s*,\s*([-+0-9.]+)\s*$', q)
        if m:
            return float(m.group(1)), float(m.group(2))
        m = re.match(r'\s*place_id\s*:\s*([A-Za-z0-9_-]+)\s*$', q)
        if m:
            return ("place_id", m.group(1))
    raise RuntimeError("Could not extract coordinates from URL.")

async def resolve_place_id_to_coords(place_id: str, api_key: str, client: httpx.AsyncClient):
    url = f"{GOOGLE_BASE}/place/details/json"
    params = {"place_id": place_id, "fields": "geometry", "key": api_key}
    try:
        r = await client.get(url, params=params, timeout=30.0)
        r.raise_for_status()
    except httpx.HTTPStatusError as e:
        logger.error(
            "Google Place Details HTTP error: status=%s url=%s params=%s body=%s",
            getattr(e.response, "status_code", "?"), url, params, getattr(e.response, "text", ""),
        )
        raise HTTPException(status_code=502, detail=f"Place Details HTTP error: {e}")
    except httpx.RequestError as e:
        logger.error("Google Place Details request error: url=%s params=%s err=%s", url, params, e)
        raise HTTPException(status_code=502, detail=f"Place Details request error: {e}")
    data = r.json()
    if data.get("status") != "OK":
        logger.error("Place Details API returned non-OK: status=%s data=%s", data.get("status"), data)
        raise HTTPException(status_code=502, detail=f"Place Details error: {data}")
    loc = data["result"]["geometry"]["location"]
    return float(loc["lat"]), float(loc["lng"])

async def fetch_static_satellite(
    lat: float,
    lng: float,
    api_key: str,
    *,
    zoom: int = 20,
    size_px: int = 640,
    scale: int = 2,
    circle_radius: int | None = None,
    client: httpx.AsyncClient | None = None,
) -> bytes:
    if size_px > 640:
        raise HTTPException(status_code=400, detail="size_px must be <= 640; use scale=2 for higher resolution.")
    params = {
        "maptype": "satellite",
        "center": f"{lat:.7f},{lng:.7f}",
        "zoom": str(zoom),
        "size": f"{size_px}x{size_px}",
        "scale": str(scale),
        "key": api_key,
    }
    if circle_radius:
        circle_path = make_circle_path(lat, lng, radius_m=circle_radius, num_points=72)
        params["path"] = f"fillcolor:0x7FFF0000|color:0xFF0000|weight:5|{circle_path}"

    close_after = False
    if client is None:
        client = httpx.AsyncClient(base_url=GOOGLE_BASE)
        close_after = True
    try:
        r = await client.get("/staticmap", params=params, timeout=30.0)
        r.raise_for_status()
        if r.headers.get("Content-Type", "").startswith("application/json"):
            logger.error("Static Maps API returned JSON error: %s", r.text)
            raise HTTPException(status_code=502, detail=f"Static Maps API error: {r.text}")
        return r.content
    except httpx.HTTPStatusError as e:
        logger.error(
            "Static Maps HTTP error: status=%s url=%s params=%s body=%s",
            getattr(e.response, "status_code", "?"), f"{GOOGLE_BASE}/staticmap", params, getattr(e.response, "text", ""),
        )
        raise HTTPException(status_code=502, detail=f"Static Maps HTTP error: {e}")
    except httpx.RequestError as e:
        logger.error("Static Maps request error: url=%s params=%s err=%s", f"{GOOGLE_BASE}/staticmap", params, e)
        raise HTTPException(status_code=502, detail=f"Static Maps request error: {e}")
    finally:
        if close_after:
            await client.aclose()

# ---------- Single endpoint: build image -> send to OpenAI -> return text ----------
@router.get("/satellite", summary="Describe a satellite image using OpenAI")
async def satellite(
    url: str | None = Query(default=None, description="Google Maps URL (supports @lat,lng, place_id, or q=...)"),
    lat: float | None = Query(default=None, description="Latitude if no URL provided"),
    lng: float | None = Query(default=None, description="Longitude if no URL provided"),
    zoom: int = Query(default=20, ge=0, le=21),
    size_px: int = Query(default=640, ge=1, le=640),
    scale: int = Query(default=2, ge=1, le=2),
    circle_radius: int | None = Query(default=None, ge=1, description="Optional circle radius (meters)"),
    preview: bool = False,
    model: str = Query(default="gpt-5", description="OpenAI model to use for analysis"),
):
    gkey = os.getenv("GOOGLE_MAPS_API_KEY")
    if not gkey:
        raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY is not configured")

    # figure out coordinates
    if url:
        try:
            parsed = extract_coords_from_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[0], float):
            use_lat, use_lng = parsed
        elif isinstance(parsed, tuple) and parsed[0] == "place_id":
            async with httpx.AsyncClient() as c:
                use_lat, use_lng = await resolve_place_id_to_coords(parsed[1], gkey, c)
        else:
            raise HTTPException(status_code=400, detail="Unrecognized URL format.")
    else:
        if lat is None or lng is None:
            raise HTTPException(status_code=400, detail="Provide either 'url' or both 'lat' and 'lng'.")
        use_lat, use_lng = float(lat), float(lng)

    # 1) get the satellite image bytes
    img_bytes = await fetch_static_satellite(
        use_lat, use_lng, gkey,
        zoom=zoom, size_px=size_px, scale=scale, circle_radius=circle_radius
    )

    if preview:
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")

    # 2) send to OpenAI and return the text
    prompt = ("""You are an expert in analyzing satellite images.
Your task is to examine the provided satellite photo and determine:
1. Whether solar panels are present.
2. Whether there are flat rooftops suitable for placing solar panels.

Instructions:
- For solar panels, look for rectangular dark-blue/black areas with a grid-like texture, often arranged in rows. They may be found on rooftops or in large open ground installations.
- For flat rooftops, look for rooftops of buildings with a flat geometry (not pitched/sloped).
- Avoid confusing solar panels with shadows, dark rooftops, or parking lots.

Provide the output strictly in the following JSON format:
{
  "solar_panels": "yes/no",
  "flat_surface": "yes/no",
  "reasoning": "short explanation of why you answered yes or no"
}""")
    text = await analyze_image_bytes_with_openai_async(img_bytes, prompt, model)
    
    # Parse the JSON response from OpenAI
    try:
        # Extract the result part if it's wrapped in a larger response
        if isinstance(text, str):
            # Try to parse as JSON first
            try:
                parsed = json.loads(text)
                # If it has a "result" key, use that; otherwise use the whole response
                if "result" in parsed:
                    result_data = json.loads(parsed["result"])
                else:
                    result_data = parsed
            except json.JSONDecodeError:
                # If the whole text isn't valid JSON, try to extract JSON from within
                # Look for JSON-like content between curly braces
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    # Fallback: return the raw text
                    result_data = {"raw_response": text}
        else:
            result_data = text
            
        return {"model": model, "result": result_data}
    except Exception as e:
        # If parsing fails, return the raw response
        return {"model": model, "result": {"raw_response": text, "parse_error": str(e)}}
