import os
import re
import math
import base64
import httpx

from io import BytesIO
from urllib.parse import urlparse, parse_qs, unquote
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query

# ---------- OpenAI ----------
from openai import OpenAI

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)

def analyze_image_bytes_with_openai(image_bytes: bytes, prompt: str) -> str:
    """
    Send a PNG (bytes) + prompt to OpenAI Responses API and return text.
    Uses a base64 data URL so no hosting is needed.
    """
    model = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
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
    r = await client.get(url, params=params, timeout=30.0)
    r.raise_for_status()
    data = r.json()
    if data.get("status") != "OK":
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
            raise HTTPException(status_code=502, detail=f"Static Maps API error: {r.text}")
        return r.content
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
    prompt = (
        "Analyze the satellite image. Focus ONLY on buildings inside the red circle and determine if there are solar panels on any roofs.\n"
        "Return ONLY a JSON object with exactly these keys and formats:\n"
        "{ \"solar_panels\": \"yes\" | \"no\", \"reasoning\": \"short explanation\" }\n"
        "Rules:\n"
        "- solar_panels must be exactly \"yes\" or \"no\".\n"
        "- reasoning should be 1-2 concise sentences.\n"
        "- Output the JSON only, with no extra text."
    )
    text = analyze_image_bytes_with_openai(img_bytes, prompt)
    return {"model": os.getenv("OPENAI_MODEL", "gpt-4o-mini"), "result": text}
