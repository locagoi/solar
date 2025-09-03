import os
import re
import math
import base64
import httpx
import json
import logging
import asyncio

from io import BytesIO
from urllib.parse import urlparse, parse_qs, unquote
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query

# ---------- OpenAI ----------
from openai import OpenAI, AsyncOpenAI

# ---------- Playwright ----------
from playwright.async_api import async_playwright

# Global browser instance - only one browser and one page for all requests
_playwright = None
_browser = None
_page = None
_browser_lock = asyncio.Lock()

# Ultra-strict concurrency limit for 512MB RAM - only 1 request at a time
playwright_semaphore = asyncio.Semaphore(1)  # Max 1 concurrent request

logger = logging.getLogger(__name__)

async def _get_shared_browser_and_page():
    """Get or create a shared browser and page instance."""
    global _playwright, _browser, _page
    
    async with _browser_lock:
        if _browser is None:
            logger.info("Creating shared browser and page instance")
            _playwright = await async_playwright().start()
            _browser = await _playwright.chromium.launch(
                headless=True,
                args=['--no-sandbox', '--disable-dev-shm-usage', '--disable-gpu']
            )
            _page = await _browser.new_page()
            # Set device scale factor only once when page is created
            await _page.evaluate("() => { Object.defineProperty(screen, 'devicePixelRatio', { get: () => 2 }); }")
            logger.info("Shared browser and page instance created")
        return _browser, _page

def _get_openai_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY is not configured")
    return OpenAI(api_key=api_key)

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

# ---------- Browserless.io satellite capture ----------
async def capture_satellite_with_browserless(
    lat: float,
    lng: float,
    *,
    zoom: int = 18,
    size_px: int = 500,
    circle_radius: int | None = None,
) -> bytes:
    """
    Capture satellite view using browserless.io service.
    This provides more flexibility than the Static Maps API without local browser dependencies.
    """
    try:
        # Get API keys from environment
        browserless_api_key = os.getenv("BROWSERLESS_API_KEY")
        google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        if not browserless_api_key:
            raise HTTPException(status_code=500, detail="BROWSERLESS_API_KEY is not configured")
        if not google_maps_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY is not configured")
        
        browserless_url = f"https://production-sfo.browserless.io/chrome/screenshot?token={browserless_api_key}"
        
        # Create HTML content with the specified coordinates
        html_content = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Satellite Map</title>
    <style>
      html, body {{ height: 100%; margin: 0; padding: 0; }}
      #map {{ width: 100vw; height: 100vh; }}
      .circle-overlay {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: {circle_radius * 2 if circle_radius else 0}px;
        height: {circle_radius * 2 if circle_radius else 0}px;
        border: 5px solid #FF0000;
        border-radius: 50%;
        background-color: rgba(255, 0, 0, 0.3);
        pointer-events: none;
        z-index: 1000;
      }}
    </style>
    <script>
      let map;
      async function initMap() {{
        const {{ Map, MapTypeId, Circle }} = await google.maps.importLibrary("maps");

        const center = {{ lat: {lat}, lng: {lng} }};
        const zoom = {zoom};

        map = new Map(document.getElementById("map"), {{
          center,
          zoom,
          mapTypeId: MapTypeId.SATELLITE,
          tilt: 0,
          heading: 0,
          disableDefaultUI: true,
        }});
        
        {f'''
        // Add circle overlay if requested
        if ({bool(circle_radius)}) {{
          new Circle({{
            strokeColor: "#FF0000",
            strokeOpacity: 1.0,
            strokeWeight: 5,
            fillColor: "#FF0000",
            fillOpacity: 0.3,
            map,
            center,
            radius: {circle_radius},
          }});
        }}
        ''' if circle_radius else ''}
      }}
      window.initMap = initMap;
    </script>
    <script async
      src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap&v=weekly">
    </script>
  </head>
  <body>
    <div id="map"></div>
    {f'<div class="circle-overlay"></div>' if circle_radius else ''}
  </body>
</html>"""
        
        payload = {
            "html": html_content,
            "waitForTimeout": 3000,  # Wait 3 seconds for map to load
            "options": {
                "type": "png",
                "fullPage": False
            },
            "viewport": {
                "width": size_px,
                "height": size_px,
                "deviceScaleFactor": 2  # Higher resolution
            }
        }
        
        # Make request to browserless.io
        async with httpx.AsyncClient() as client:
            response = await client.post(browserless_url, json=payload, timeout=60.0)
            response.raise_for_status()
            
            if response.headers.get("Content-Type", "").startswith("image/"):
                return response.content
            else:
                logger.error("Browserless returned non-image content: %s", response.text)
                raise HTTPException(status_code=502, detail="Browserless returned non-image content")
                
    except httpx.HTTPStatusError as e:
        logger.error("Browserless HTTP error: status=%s body=%s", e.response.status_code, e.response.text)
        raise HTTPException(status_code=502, detail=f"Browserless HTTP error: {e}")
    except httpx.RequestError as e:
        logger.error("Browserless request error: %s", e)
        raise HTTPException(status_code=502, detail=f"Browserless request error: {e}")
    except Exception as e:
        logger.exception("Browserless capture failed: %s", e)
        raise HTTPException(status_code=502, detail=f"Browserless capture error: {e}")

# ---------- Playwright satellite capture ----------
async def capture_satellite_with_playwright(
    lat: float,
    lng: float,
    *,
    zoom: int = 18,
    size_px: int = 500,
    circle_radius: int | None = None,
) -> bytes:
    """
    Capture satellite view using Playwright browser automation.
    This provides local browser control without external service dependencies.
    """
    start_time = asyncio.get_event_loop().time()
    try:
        # Get Google Maps API key from environment
        google_maps_api_key = os.getenv("GOOGLE_MAPS_API_KEY")
        
        if not google_maps_api_key:
            raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY is not configured")
        
        # Create HTML content with the specified coordinates
        html_content = f"""<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Satellite Map</title>
    <style>
      html, body {{ height: 100%; margin: 0; padding: 0; }}
      #map {{ width: 100vw; height: 100vh; }}
      .circle-overlay {{
        position: absolute;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%);
        width: {circle_radius * 2 if circle_radius else 0}px;
        height: {circle_radius * 2 if circle_radius else 0}px;
        border: 5px solid #FF0000;
        border-radius: 50%;
        background-color: rgba(255, 0, 0, 0.3);
        pointer-events: none;
        z-index: 1000;
      }}
    </style>
    <script>
      let map;
      async function initMap() {{
        const {{ Map, MapTypeId, Circle }} = await google.maps.importLibrary("maps");

        const center = {{ lat: {lat}, lng: {lng} }};
        const zoom = {zoom};

        map = new Map(document.getElementById("map"), {{
          center,
          zoom,
          mapTypeId: MapTypeId.SATELLITE,
          tilt: 0,
          heading: 0,
          disableDefaultUI: true,
        }});
        
        {f'''
        // Add circle overlay if requested
        if ({bool(circle_radius)}) {{
          new Circle({{
            strokeColor: "#FF0000",
            strokeOpacity: 1.0,
            strokeWeight: 5,
            fillColor: "#FF0000",
            fillOpacity: 0.3,
            map,
            center,
            radius: {circle_radius},
          }});
        }}
        ''' if circle_radius else ''}
      }}
      window.initMap = initMap;
    </script>
    <script async
      src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap&v=weekly">
    </script>
  </head>
  <body>
    <div id="map"></div>
    {f'<div class="circle-overlay"></div>' if circle_radius else ''}
  </body>
</html>"""
        
        # Use shared browser and page instance with strict concurrency limit
        async with playwright_semaphore:
            browser, page = await _get_shared_browser_and_page()
            
            # Reuse the same page for each request
            await page.set_viewport_size({"width": size_px, "height": size_px})
            
            # Clear page state to prevent memory accumulation
            await page.evaluate("() => { window.map = null; }")
            await page.goto("about:blank")  # Clear the page completely
            
            await page.set_content(html_content)
            
            # Wait for the map to load (wait for the map div to be populated)
            await page.wait_for_function(
                "() => window.map && document.querySelector('#map').children.length > 0",
                timeout=10000
            )
            
            # Additional wait to ensure tiles are loaded
            await page.wait_for_timeout(3000)
            
            # Take screenshot
            screenshot_bytes = await page.screenshot(
                type="png",
                full_page=False,
                clip={"x": 0, "y": 0, "width": size_px, "height": size_px}
            )
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(f"Playwright success: {duration:.1f}s, {len(screenshot_bytes)} bytes")
            return screenshot_bytes
                
    except Exception as e:
        duration = asyncio.get_event_loop().time() - start_time
        logger.error(f"Playwright failed after {duration:.1f}s: {type(e).__name__}: {str(e)}")
        raise HTTPException(status_code=502, detail=f"Playwright capture error: {e}")

# ---------- Single endpoint: build image -> send to OpenAI -> return text ----------
@router.get("/satellite", summary="Capture satellite image using browserless.io service")
async def satellite(
    url: str | None = Query(default=None, description="Google Maps URL (supports @lat,lng, place_id, or q=...)"),
    lat: float | None = Query(default=None, description="Latitude if no URL provided"),
    lng: float | None = Query(default=None, description="Longitude if no URL provided"),
    zoom: int = Query(default=18, ge=0, le=21),
    size_px: int = Query(default=500, ge=1, le=1280),
    circle_radius: int | None = Query(default=None, ge=1, description="Optional circle radius (meters)"),
    preview: bool = False,
    model: str = Query(default="gpt-5", description="OpenAI model to use for analysis"),
):
    """
    Capture satellite view using browserless.io service.
    This provides more flexibility than the Static Maps API without local browser dependencies.
    """
    # figure out coordinates (reuse existing logic)
    if url:
        try:
            parsed = extract_coords_from_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[0], float):
            use_lat, use_lng = parsed
        elif isinstance(parsed, tuple) and parsed[0] == "place_id":
            # For place_id, we still need Google Maps API key
            gkey = os.getenv("GOOGLE_MAPS_API_KEY")
            if not gkey:
                raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY is required for place_id resolution")
            async with httpx.AsyncClient() as c:
                use_lat, use_lng = await resolve_place_id_to_coords(parsed[1], gkey, c)
        else:
            raise HTTPException(status_code=400, detail="Unrecognized URL format.")
    else:
        if lat is None or lng is None:
            raise HTTPException(status_code=400, detail="Provide either 'url' or both 'lat' and 'lng'.")
        use_lat, use_lng = float(lat), float(lng)

    # Capture satellite image using browserless
    img_bytes = await capture_satellite_with_browserless(
        use_lat, use_lng,
        zoom=zoom, size_px=size_px, circle_radius=circle_radius
    )

    if preview:
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")

    # Send to OpenAI and return the text (reuse existing prompt)
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
    
    # Parse the JSON response from OpenAI (reuse existing logic)
    try:
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if "result" in parsed:
                    result_data = json.loads(parsed["result"])
                else:
                    result_data = parsed
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    result_data = {"raw_response": text}
        else:
            result_data = text
            
        return {"model": model, "result": result_data}
    except Exception as e:
        return {"model": model, "result": {"raw_response": text, "parse_error": str(e)}}

@router.get("/satellite-playwright", summary="Capture satellite image using Playwright browser automation")
async def satellite_playwright(
    url: str | None = Query(default=None, description="Google Maps URL (supports @lat,lng, place_id, or q=...)"),
    lat: float | None = Query(default=None, description="Latitude if no URL provided"),
    lng: float | None = Query(default=None, description="Longitude if no URL provided"),
    zoom: int = Query(default=18, ge=0, le=21),
    size_px: int = Query(default=1000, ge=1, le=1280),
    circle_radius: int | None = Query(default=None, ge=1, description="Optional circle radius (meters)"),
    preview: bool = False,
    model: str = Query(default="gpt-5", description="OpenAI model to use for analysis"),
):
    """
    Capture satellite view using Playwright browser automation.
    This provides local browser control without external service dependencies.
    """
    # figure out coordinates (reuse existing logic)
    if url:
        try:
            parsed = extract_coords_from_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[0], float):
            use_lat, use_lng = parsed
        elif isinstance(parsed, tuple) and parsed[0] == "place_id":
            # For place_id, we still need Google Maps API key
            gkey = os.getenv("GOOGLE_MAPS_API_KEY")
            if not gkey:
                raise HTTPException(status_code=500, detail="GOOGLE_MAPS_API_KEY is required for place_id resolution")
            async with httpx.AsyncClient() as c:
                use_lat, use_lng = await resolve_place_id_to_coords(parsed[1], gkey, c)
        else:
            raise HTTPException(status_code=400, detail="Unrecognized URL format.")
    else:
        if lat is None or lng is None:
            raise HTTPException(status_code=400, detail="Provide either 'url' or both 'lat' and 'lng'.")
        use_lat, use_lng = float(lat), float(lng)

    # Capture satellite image using Playwright
    img_bytes = await capture_satellite_with_playwright(
        use_lat, use_lng,
        zoom=zoom, size_px=size_px, circle_radius=circle_radius
    )

    if preview:
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")

    # Send to OpenAI and return the text (reuse existing prompt)
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
    
    # Parse the JSON response from OpenAI (reuse existing logic)
    try:
        if isinstance(text, str):
            try:
                parsed = json.loads(text)
                if "result" in parsed:
                    result_data = json.loads(parsed["result"])
                else:
                    result_data = parsed
            except json.JSONDecodeError:
                json_match = re.search(r'\{.*\}', text, re.DOTALL)
                if json_match:
                    result_data = json.loads(json_match.group())
                else:
                    result_data = {"raw_response": text}
        else:
            result_data = text
            
        return {"model": model, "result": result_data}
    except Exception as e:
        return {"model": model, "result": {"raw_response": text, "parse_error": str(e)}}

