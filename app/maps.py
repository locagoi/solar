import os
import re
import math
import base64
import httpx
import json
import logging
import asyncio
import gc

from io import BytesIO
from urllib.parse import urlparse, parse_qs, unquote
from fastapi.responses import StreamingResponse
from fastapi import APIRouter, HTTPException, Query, Depends, Header

# ---------- OpenAI ----------
from openai import OpenAI, AsyncOpenAI

# ---------- Playwright ----------
from playwright.async_api import async_playwright

# ---------- Supabase ----------
from supabase import create_client, Client

# Global browser instance - only one browser and one page for all requests
_playwright = None
_browser = None
_page = None
_browser_lock = asyncio.Lock()

# Ultra-strict concurrency limit for 512MB RAM - only 1 request at a time
playwright_semaphore = asyncio.Semaphore(1)  # Max 1 concurrent request

logger = logging.getLogger(__name__)

# ---------- Authentication ----------
async def verify_bearer_token(authorization: str = Header(None)):
    """Verify bearer token from Authorization header."""
    if not authorization:
        raise HTTPException(status_code=401, detail="Authorization header missing")
    
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header format")
    
    token = authorization[7:]  # Remove "Bearer " prefix
    
    expected_token = os.getenv("BEARER_TOKEN")
    if not expected_token:
        raise HTTPException(status_code=500, detail="BEARER_TOKEN is not configured")
    
    if token != expected_token:
        raise HTTPException(status_code=401, detail="Invalid bearer token")
    
    return token

# ---------- Supabase Storage ----------
def get_supabase_client() -> Client:
    """Get Supabase client instance with service key (bypasses RLS)."""
    supabase_url = os.getenv("SUPABASE_URL")
    supabase_service_key = os.getenv("SUPABASE_SERVICE_KEY")
    
    if not supabase_url:
        raise HTTPException(status_code=500, detail="SUPABASE_URL is not configured")
    if not supabase_service_key:
        raise HTTPException(status_code=500, detail="SUPABASE_SERVICE_KEY is not configured")
    
    return create_client(supabase_url, supabase_service_key)

async def upload_image_to_supabase(
    image_bytes: bytes,
    filename: str,
    bucket_name: str = "photos"
) -> str:
    """
    Upload image bytes to Supabase storage and return the public URL.
    
    Args:
        image_bytes: The image data as bytes
        filename: Name for the uploaded file
        bucket_name: Supabase storage bucket name
        
    Returns:
        Public URL of the uploaded image
    """
    try:
        supabase = get_supabase_client()
        
        # Check if file exists and delete it first to ensure overwrite
        try:
            supabase.storage.from_(bucket_name).remove([filename])
        except Exception:
            # File doesn't exist, which is fine - we'll create it
            pass
        
        # Upload the image to Supabase storage
        response = supabase.storage.from_(bucket_name).upload(
            path=filename,
            file=image_bytes,
            file_options={"content-type": "image/png"}
        )
        
        # Check if upload was successful
        if hasattr(response, 'status_code') and response.status_code >= 400:
            logger.error(f"Supabase upload failed with status: {response.status_code}")
            raise HTTPException(status_code=502, detail=f"Supabase upload failed with status: {response.status_code}")
        
        # Get the public URL
        public_url = supabase.storage.from_(bucket_name).get_public_url(filename)
        
        logger.info(f"Image uploaded to Supabase: {public_url}")
        return public_url
        
    except Exception as e:
        logger.exception(f"Supabase upload failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to upload image to Supabase: {str(e)}")

async def save_analysis_to_meta(
    photo_name: str,
    solar_panels: bool,
    flat_surface: bool,
    reasoning: str,
    place_id: str,
    suitability_score: int = None,
    suitability_reasoning: str = None
) -> dict:
    """
    Save satellite analysis results to the meta table.
    Updates existing record if photo_name exists, otherwise creates new record.
    
    Args:
        photo_name: Name of the photo file
        solar_panels: Whether solar panels are present
        flat_surface: Whether flat surface is suitable
        reasoning: Explanation of the analysis
        place_id: Google Place ID (required)
        suitability_score: Solar panel suitability score (0-100)
        suitability_reasoning: Detailed suitability analysis reasoning
        
    Returns:
        Dictionary with the saved/updated record data
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare the data for upsert
        data = {
            "photo_name": photo_name,
            "solar_panels": solar_panels,
            "flat_surface": flat_surface,
            "reasoning": reasoning,
            "place_id": place_id,
            "updated_at": "now()"
        }
        
        # Add suitability data if provided
        if suitability_score is not None:
            data["suitability_score"] = suitability_score
        if suitability_reasoning is not None:
            data["suitability_reasoning"] = suitability_reasoning
        
        # Use upsert to update if exists, insert if not
        # This will update the record if photo_name already exists
        response = supabase.table("meta").upsert(
            data, 
            on_conflict="photo_name"
        ).execute()
        
        if hasattr(response, 'data') and response.data:
            return response.data[0]
        else:
            logger.error(f"Failed to upsert to meta table: {response}")
            raise HTTPException(status_code=502, detail="Failed to save analysis to meta table")
            
    except Exception as e:
        logger.exception(f"Meta table upsert failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to save analysis to meta table: {str(e)}")

async def save_leads_data(
    company_name: str = None,
    website: str = None,
    phone: str = None,
    address: str = None,
    country: str = None,
    person_first_name: str = None,
    person_last_name: str = None,
    person_phone: str = None,
    person_email: str = None,
    google_maps_url: str = None,
    place_id: str = None
) -> dict:
    """
    Save company and person leads data to the leads table.
    Updates existing record if place_id exists, otherwise creates new record.
    
    Returns:
        Dictionary with the saved/updated record data
    """
    try:
        supabase = get_supabase_client()
        
        # Prepare the data for upsert
        data = {
            "company_name": company_name,
            "website": website,
            "phone": phone,
            "address": address,
            "country": country,
            "person_first_name": person_first_name,
            "person_last_name": person_last_name,
            "person_phone": person_phone,
            "person_email": person_email,
            "google_maps_url": google_maps_url,
            "place_id": place_id,
            "updated_at": "now()"
        }
        
        # Remove None values to avoid storing empty strings
        data = {k: v for k, v in data.items() if v is not None}
        
        # Use upsert to update if exists, insert if not
        # This will update the record if place_id already exists
        response = supabase.table("leads").upsert(
            data, 
            on_conflict="place_id"
        ).execute()
        
        if hasattr(response, 'data') and response.data:
            return response.data[0]
        else:
            logger.error(f"Failed to upsert to leads table: {response}")
            raise HTTPException(status_code=502, detail="Failed to save leads data")
            
    except Exception as e:
        logger.exception(f"Leads table upsert failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to save leads data: {str(e)}")

def _log_memory(step: str):
    """Log current memory usage for debugging."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        memory_info = process.memory_info()
        rss_mb = round(memory_info.rss / 1024 / 1024, 2)
        vms_mb = round(memory_info.vms / 1024 / 1024, 2)
        percent = round(process.memory_percent(), 2)
        logger.info(f"Memory [{step}]: RSS={rss_mb}MB, VMS={vms_mb}MB, {percent}%")
    except Exception as e:
        logger.warning(f"Memory logging failed: {e}")

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

async def analyze_suitability_with_openai_async(image_bytes: bytes, prompt: str, model: str = "gpt-5") -> str:
    """
    Async variant for suitability analysis that avoids blocking the event loop during OpenAI network I/O.
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
        logger.exception("OpenAI suitability analysis request failed: %s", e)
        raise HTTPException(status_code=502, detail=f"OpenAI suitability analysis error: {e}")

# ---------- Google Maps ----------
router = APIRouter(prefix="/maps", tags=["maps"])
GOOGLE_BASE = "https://maps.googleapis.com/maps/api"

# OpenAI prompt for satellite image analysis
SATELLITE_ANALYSIS_PROMPT = """You are an expert in analyzing satellite images.
Your task is to examine the provided satellite photo and determine:
1. Whether solar panels are present.
2. Whether there are flat rooftops suitable for placing solar panels.

Instructions:
- For solar panels, look for rectangular dark-blue/black areas with a grid-like texture, often arranged in rows. They may be found on rooftops or in large open ground installations.
- For flat rooftops suitable for solar panels, look for rooftops of buildings with flat or nearly flat geometry (not pitched/sloped). A suitable roof should be:
  - Large enough to hold multiple solar panels.
  - Mostly unobstructed by large structures (e.g., trees, towers, tall rooftop equipment).
  - Having a reasonably consistent surface, where small skylights or HVAC units are acceptable.
- Avoid confusing solar panels with shadows, dark rooftops, or parking lots.
- Do not classify parking lots, ground surfaces, or sloped roofs as suitable flat rooftops.

Provide the output strictly in the following JSON format:
{
  "solar_panels": true/false,
  "flat_surface": true/false,
  "reasoning": "short explanation of why you answered true or false"
}"""

# OpenAI prompt for solar panel suitability analysis
SUITABILITY_ANALYSIS_PROMPT = """You are an expert in analyzing solar panel suitability for rooftops.
Your task is to examine the provided satellite photo and determine the solar panel suitability of the property located in the center of the image.

Analyze the roof(s) of the central property according to the following scoring system:

Scoring form (image-based only, 0–100 points):

Usable Roof Area (0–30)
- 1000 m² → 30
- 500–1000 m² → 24
- 200–499 m² → 16
- 100–199 m² → 8
- <100 m² → 0 (Reject)

Shading (0–30)
- No visible obstacles South/East/West → 30
- Few small obstacles, distance >1.5× their height → 22
- Several obstacles, distance ≈1× height → 14
- Dense trees/buildings close to roof → 6
- Heavy shading → 0 (Reject)

Obstructions & Shape (0–20)
- <5% of roof blocked (skylights, HVAC, chimneys) → 20
- 5–10% → 15
- 10–20% → 10
- 20% → 0 (Reject)

Orientation & Slope (0–10)
- South-facing, 10–35° tilt → 10
- Flat roof (can be tilted) → 8
- East/West → 7
- North/steep tilt → 3

Accessibility (0–10)    
- Good access/clear crane position visible → 10
- Limited access → 5
- Very restricted/no access → 0

Total Score = Sum (max. 100)

Classification:
A-Roof: ≥75 points → very attractive
B-Roof: 60–74 → good
C-Roof: 45–59 → borderline
Reject: <45 or KO

Output format:
Return the result strictly in the following JSON format (no extra text):
{
  "score": 0-100,
  "reasoning": "Detailed explanation of how you scored each category and why"
}"""

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


# ---------- Playwright satellite capture ----------
async def capture_satellite_with_playwright(
    lat: float,
    lng: float,
    *,
    zoom: int = 18,
    size_px: int = 500,
) -> bytes:
    """
    Capture satellite view using Playwright browser automation.
    This provides local browser control without external service dependencies.
    """
    start_time = asyncio.get_event_loop().time()
    _log_memory("START")
    
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
    </style>
    <script>
      var map;
      async function initMap() {{
        const {{ Map, MapTypeId }} = await google.maps.importLibrary("maps");

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
        
        // Set window.map for reliable detection
        window.map = map;
      }}
      window.initMap = initMap;
    </script>
    <script async
      src="https://maps.googleapis.com/maps/api/js?key={google_maps_api_key}&callback=initMap&v=weekly">
    </script>
  </head>
  <body>
    <div id="map"></div>
  </body>
</html>"""

    async with playwright_semaphore:
        browser, _ = await _get_shared_browser_and_page()  # keep single browser

        # NEW: short-lived, incognito context & page per request
        context = await browser.new_context(
            viewport={"width": size_px, "height": size_px},
            device_scale_factor=1,  # sharper output
        )
        page = await context.new_page()
        
        try:
            await page.set_content(html_content, wait_until="domcontentloaded", timeout=15000)

            # FIX: Wait for tiles using DOM-based signal
            await page.wait_for_function(
                "() => document.querySelector('#map') && "
                "document.querySelector('#map').children.length > 0",
                timeout=10000
            )
            
            await page.wait_for_timeout(1500)  # small extra settle

            # Take screenshot
            screenshot_bytes = await page.screenshot(
                type="png",
                full_page=False,
                clip={"x": 0, "y": 0, "width": size_px, "height": size_px}
            )
            
            # Python-level memory cleanup
            gc.collect()  # Force Python garbage collection
            
            duration = asyncio.get_event_loop().time() - start_time
            logger.info(f"Playwright success: {duration:.1f}s, {len(screenshot_bytes)} bytes")
            return screenshot_bytes
            
        finally:
            # CRITICAL: free resources deterministically
            try:
                await page.close()
            except:  # noqa
                pass
            try:
                await context.close()
            except:  # noqa
                pass


@router.get("/satellite", summary="Capture satellite image using Playwright browser automation")
async def satellite(
    url: str | None = Query(default=None, description="Google Maps URL (supports @lat,lng, place_id, or q=...)"),
    lat: float | None = Query(default=None, description="Latitude if no URL provided"),
    lng: float | None = Query(default=None, description="Longitude if no URL provided"),
    zoom: int = Query(default=18, ge=0, le=21),
    size_px: int = Query(default=1000, ge=1, le=1280),
    preview: bool = False,
    model: str = Query(default="gpt-5", description="OpenAI model to use for analysis"),
    suitability: bool = Query(default=False, description="Analyze solar panel suitability"),
    token: str = Depends(verify_bearer_token),
):
    """
    Capture satellite view using Playwright browser automation.
    This provides local browser control without external service dependencies.
    """
    # figure out coordinates (reuse existing logic)
    place_id = None
    if url:
        try:
            parsed = extract_coords_from_url(url)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        if isinstance(parsed, tuple) and len(parsed) == 2 and isinstance(parsed[0], float):
            use_lat, use_lng = parsed
        elif isinstance(parsed, tuple) and parsed[0] == "place_id":
            # For place_id, we still need Google Maps API key
            place_id = parsed[1]  # Store the place_id for filename
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
    try:
        img_bytes = await capture_satellite_with_playwright(
            use_lat, use_lng,
            zoom=zoom, size_px=size_px
        )
        
        # Validate image bytes
        if not img_bytes or len(img_bytes) == 0:
            raise HTTPException(status_code=502, detail="Failed to capture satellite image: empty response")
            
    except Exception as e:
        logger.error(f"Satellite image capture failed: {e}")
        raise HTTPException(status_code=502, detail=f"Failed to capture satellite image: {str(e)}")

    # Always upload to Supabase
    supabase_url = None
    try:
        import datetime
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Use place_id as filename if available, otherwise use coordinates
        if place_id:
            filename = f"{place_id}.png"
        else:
            filename = f"satellite_{use_lat:.6f}_{use_lng:.6f}.png"
            
        supabase_url = await upload_image_to_supabase(img_bytes, filename)
    except Exception as e:
        logger.error(f"Supabase upload failed: {e}")
        # Don't fail the entire request if upload fails, just log the error

    if preview:
        return StreamingResponse(BytesIO(img_bytes), media_type="image/png")

    # Send to OpenAI and return the text
    text = await analyze_image_bytes_with_openai_async(img_bytes, SATELLITE_ANALYSIS_PROMPT, model)
    
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
            
        # Perform suitability analysis if requested
        suitability_data = None
        if suitability:
            try:
                suitability_text = await analyze_suitability_with_openai_async(img_bytes, SUITABILITY_ANALYSIS_PROMPT, model)
                if isinstance(suitability_text, str):
                    try:
                        suitability_parsed = json.loads(suitability_text)
                        if "result" in suitability_parsed:
                            suitability_data = json.loads(suitability_parsed["result"])
                        else:
                            suitability_data = suitability_parsed
                    except json.JSONDecodeError:
                        suitability_json_match = re.search(r'\{.*\}', suitability_text, re.DOTALL)
                        if suitability_json_match:
                            suitability_data = json.loads(suitability_json_match.group())
                        else:
                            suitability_data = {"raw_response": suitability_text}
                else:
                    suitability_data = suitability_text
            except Exception as e:
                logger.error(f"Suitability analysis failed: {e}")
                suitability_data = {"error": str(e)}
            
        # Save analysis results to meta table if we have valid data
        meta_record = None
        if isinstance(result_data, dict) and all(key in result_data for key in ["solar_panels", "flat_surface", "reasoning"]):
            try:
                # Convert string values to boolean if needed
                solar_panels = result_data["solar_panels"]
                flat_surface = result_data["flat_surface"]
                
                if isinstance(solar_panels, str):
                    solar_panels = solar_panels.lower() in ["true", "yes"]
                if isinstance(flat_surface, str):
                    flat_surface = flat_surface.lower() in ["true", "yes"]
                
                # Extract suitability data if available
                suitability_score = None
                suitability_reasoning = None
                if suitability_data and isinstance(suitability_data, dict):
                    if "score" in suitability_data:
                        try:
                            suitability_score = int(suitability_data["score"])
                        except (ValueError, TypeError):
                            logger.warning(f"Could not convert suitability score to int: {suitability_data.get('score')}")
                    if "reasoning" in suitability_data:
                        suitability_reasoning = suitability_data["reasoning"]
                
                meta_record = await save_analysis_to_meta(
                    photo_name=filename,
                    solar_panels=solar_panels,
                    flat_surface=flat_surface,
                    reasoning=result_data["reasoning"],
                    place_id=place_id,
                    suitability_score=suitability_score,
                    suitability_reasoning=suitability_reasoning
                )
            except Exception as e:
                logger.error(f"Failed to save to meta table: {e}")
                # Don't fail the entire request if meta save fails
        
        response_data = {"model": model, "result": result_data}
        if supabase_url:
            response_data["supabase_url"] = supabase_url
        if meta_record:
            response_data["meta_id"] = meta_record["id"]
        if suitability_data:
            response_data["suitability"] = suitability_data
        return response_data
    except Exception as e:
        response_data = {"model": model, "result": {"raw_response": text, "parse_error": str(e)}}
        if supabase_url:
            response_data["supabase_url"] = supabase_url
        return response_data

@router.post("/leads", summary="Save company and person leads data")
async def save_leads(
    company_name: str = Query(default=None, description="Company name"),
    website: str = Query(default=None, description="Company website"),
    phone: str = Query(default=None, description="Company phone number"),
    address: str = Query(default=None, description="Company address"),
    country: str = Query(default=None, description="Country"),
    person_first_name: str = Query(default=None, description="Contact person first name"),
    person_last_name: str = Query(default=None, description="Contact person last name"),
    person_phone: str = Query(default=None, description="Contact person phone"),
    person_email: str = Query(default=None, description="Contact person email"),
    google_maps_url: str = Query(description="Google Maps URL (required for place_id extraction)"),
    token: str = Depends(verify_bearer_token),
):
    """
    Save company and person leads data to the leads table.
    Updates existing record if place_id exists, otherwise creates new record.
    """
    try:
        # Extract place_id from google_maps_url
        if not google_maps_url:
            raise HTTPException(status_code=400, detail="google_maps_url is required")
        
        try:
            parsed = extract_coords_from_url(google_maps_url)
            if isinstance(parsed, tuple) and parsed[0] == "place_id":
                place_id = parsed[1]
            else:
                raise HTTPException(status_code=400, detail="Could not extract place_id from Google Maps URL")
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid Google Maps URL: {str(e)}")
        
        leads_record = await save_leads_data(
            company_name=company_name,
            website=website,
            phone=phone,
            address=address,
            country=country,
            person_first_name=person_first_name,
            person_last_name=person_last_name,
            person_phone=person_phone,
            person_email=person_email,
            google_maps_url=google_maps_url,
            place_id=place_id
        )
        
        return {
            "success": True,
            "message": "Leads data saved successfully",
            "leads_id": leads_record["id"],
            "data": leads_record
        }
        
    except Exception as e:
        logger.error(f"Failed to save leads data: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to save leads data: {str(e)}")

