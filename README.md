# Solar - Satellite Image Analysis for Solar Panel Suitability

A FastAPI-based web service that analyzes satellite imagery to assess solar panel suitability for properties. The system automatically captures satellite images from Google Maps and uses OpenAI vision models to evaluate rooftops for solar installation potential.

## Overview

**Solar** is designed for solar installation companies and lead generation platforms that need to automatically assess properties for solar potential. It combines browser automation, AI vision analysis, and cloud storage to provide comprehensive rooftop analysis at scale.

## Core Features

### 1. Satellite Image Capture
- Captures high-resolution satellite views from Google Maps using Playwright browser automation
- Supports multiple input formats:
  - Direct coordinates (latitude/longitude)
  - Google Maps URLs (with coordinate extraction)
  - Google Place IDs (with automatic coordinate resolution)
- Configurable zoom levels (0-21) and image sizes (up to 1280px)

### 2. AI-Powered Analysis

**Basic Analysis:**
- Detects existing solar panels on properties
- Identifies flat rooftops suitable for solar panel installation
- Provides reasoning for each determination

**Advanced Suitability Scoring (Optional):**
Comprehensive 0-100 scoring system based on:
- **Usable Roof Area** (0-30 points): Evaluates available space for panels
- **Shading** (0-30 points): Assesses obstacles and shadow impact
- **Obstructions & Shape** (0-20 points): Analyzes roof geometry and obstacles
- **Orientation & Slope** (0-10 points): Evaluates roof angle and direction
- **Accessibility** (0-10 points): Assesses installation access

**Classification:**
- **A-Roof**: ≥75 points (very attractive)
- **B-Roof**: 60-74 points (good)
- **C-Roof**: 45-59 points (borderline)
- **Reject**: <45 points

### 3. Data Management
- Automatic image upload to Supabase Storage
- Analysis results stored in `meta` table
- Lead management system with `leads` table for company/person data
- Upsert operations to prevent duplicates

## Technical Architecture

### Technology Stack
- **FastAPI** - Modern, fast web framework for building APIs
- **Playwright** - Browser automation for satellite image capture
- **OpenAI API** - Vision model analysis (default: `gpt-5`)
- **Supabase** - Cloud storage and PostgreSQL database
- **Google Maps API** - Satellite imagery source
- **Gunicorn + Uvicorn** - Production ASGI server
- **Python 3.12** - Runtime environment

### Performance Optimizations

1. **Memory Management**
   - Shared browser instance across requests (single worker)
   - Isolated browser contexts per request (~50MB each)
   - Semaphore limiting concurrent screenshots (3 concurrent by default)
   - Automatic worker restarts after 100 requests to prevent memory leaks

2. **Async Operations**
   - Parallel execution of Supabase upload, main analysis, and suitability analysis
   - Non-blocking async OpenAI API calls
   - Concurrent I/O operations to reduce total response time

3. **Resource Cleanup**
   - Deterministic cleanup of browser contexts and pages
   - Python garbage collection after operations
   - Proper exception handling to ensure resources are freed

## API Endpoints

### `GET /maps/satellite`
Main endpoint for satellite image capture and analysis.

**Parameters:**
- `url` (optional): Google Maps URL (supports @lat,lng, place_id, or q=...)
- `lat` (optional): Latitude if no URL provided
- `lng` (optional): Longitude if no URL provided
- `zoom` (default: 18): Zoom level (0-21)
- `size_px` (default: 1000): Image size in pixels (1-1280)
- `preview` (default: false): Return image directly instead of analysis
- `model` (default: "gpt-5"): OpenAI model to use for analysis
- `suitability` (default: false): Enable suitability scoring analysis

**Authentication:** Bearer token required

**Response:**
```json
{
  "model": "gpt-5",
  "result": {
    "solar_panels": true/false,
    "flat_surface": true/false,
    "reasoning": "explanation..."
  },
  "coordinates": {
    "lat": 40.7128,
    "lng": -74.0060
  },
  "token_usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "total_tokens": 1500
  },
  "suitability": {
    "score": 85,
    "reasoning": "detailed scoring explanation..."
  }
}
```

### `GET /maps/satellite/custom`
Custom satellite image analysis endpoint with user-provided prompt and schema.

**Parameters:**
- `prompt` (required): Custom analysis prompt text
- `schema` (required): JSON schema string for structured output format
- `url` (optional): Google Maps URL (supports @lat,lng, place_id, or q=...)
- `lat` (optional): Latitude if no URL provided
- `lng` (optional): Longitude if no URL provided
- `zoom` (default: 18): Zoom level (0-21)
- `size_px` (default: 1000): Image size in pixels (1-1280)
- `preview` (default: false): Return image directly instead of analysis
- `model` (default: "gpt-5"): OpenAI model to use for analysis

**Authentication:** Bearer token required

**Response:**
```json
{
  "model": "gpt-5",
  "result": {
    // Structure matches the provided schema
  },
  "coordinates": {
    "lat": 40.7128,
    "lng": -74.0060
  },
  "token_usage": {
    "prompt_tokens": 1000,
    "completion_tokens": 500,
    "total_tokens": 1500
  }
}
```

**Example:**
```bash
curl -X GET "http://localhost:8000/maps/satellite/custom?lat=40.7128&lng=-74.0060&prompt=Analyze%20this%20satellite%20image&schema=%7B%22type%22%3A%22object%22%2C%22properties%22%3A%7B%22analysis%22%3A%7B%22type%22%3A%22string%22%7D%7D%7D" \
  -H "Authorization: Bearer your_token"
```

### `POST /maps/leads`
Save company and person leads data.

**Parameters:**
- `company_name`: Company name
- `website`: Company website
- `phone`: Company phone number
- `address`: Company address
- `country`: Country
- `person_first_name`: Contact person first name
- `person_last_name`: Contact person last name
- `person_phone`: Contact person phone
- `person_email`: Contact person email
- `google_maps_url`: Google Maps URL (required for place_id extraction)

**Authentication:** Bearer token required

### `GET /health`
Health check endpoint.

**Response:**
```json
{
  "status": "ok"
}
```

## Setup & Installation

### Prerequisites
- Python 3.12+
- Docker (for containerized deployment)
- API Keys:
  - OpenAI API key
  - Google Maps API key
  - Supabase URL and service key
  - Bearer token for API authentication

### Environment Variables

Create a `.env` file with the following variables:

```env
# OpenAI
OPENAI_API_KEY=your_openai_api_key

# Google Maps
GOOGLE_MAPS_API_KEY=your_google_maps_api_key

# Supabase
SUPABASE_URL=your_supabase_url
SUPABASE_SERVICE_KEY=your_supabase_service_key

# Authentication
BEARER_TOKEN=your_bearer_token

# Logging
LOG_LEVEL=INFO
```

### Docker Deployment

1. Build the Docker image:
```bash
docker build -t solar-app .
```

2. Run the container:
```bash
docker run -d \
  -p 8000:8000 \
  --env-file .env \
  --name solar \
  solar-app
```

### Local Development

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Install Playwright browsers:
```bash
playwright install chromium
```

3. Run with development script:
```bash
# Windows
run-dev.bat

# Linux/Mac
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage Examples

### Basic Analysis
```bash
curl -X GET "http://localhost:8000/maps/satellite?lat=40.7128&lng=-74.0060&zoom=20" \
  -H "Authorization: Bearer your_token"
```

### Analysis with Suitability Scoring
```bash
curl -X GET "http://localhost:8000/maps/satellite?url=https://maps.google.com/...&suitability=true" \
  -H "Authorization: Bearer your_token"
```

### Using Place ID
```bash
curl -X GET "http://localhost:8000/maps/satellite?url=https://maps.google.com/?q=place_id:ChIJN1t_tDeuEmsRUsoyG83frY4" \
  -H "Authorization: Bearer your_token"
```

### Save Leads Data
```bash
curl -X POST "http://localhost:8000/maps/leads?google_maps_url=https://maps.google.com/...&company_name=Example Corp&person_email=contact@example.com" \
  -H "Authorization: Bearer your_token"
```

## Database Schema

### `meta` Table
Stores satellite analysis results:
- `photo_name`: Filename of the satellite image
- `solar_panels`: Boolean indicating if solar panels are present
- `flat_surface`: Boolean indicating if flat surface is suitable
- `reasoning`: Explanation of the analysis
- `place_id`: Google Place ID
- `suitability_score`: Solar panel suitability score (0-100)
- `suitability_reasoning`: Detailed suitability analysis
- `updated_at`: Timestamp

### `leads` Table
Stores company and person lead information:
- `company_name`: Company name
- `website`: Company website
- `phone`: Company phone
- `address`: Company address
- `country`: Country
- `person_first_name`: Contact person first name
- `person_last_name`: Contact person last name
- `person_phone`: Contact person phone
- `person_email`: Contact person email
- `google_maps_url`: Google Maps URL
- `place_id`: Google Place ID (unique)
- `updated_at`: Timestamp

## Security

- Bearer token authentication required for all endpoints
- Environment variables for sensitive credentials
- Non-root user in Docker container
- Service key authentication for Supabase (bypasses RLS)

## Performance Considerations

- **Memory**: Optimized for ~512MB-2GB RAM environments
- **Concurrency**: Supports 3 concurrent screenshot operations by default
- **Response Time**: Typically 5-15 seconds per analysis (depending on OpenAI API)
- **Scalability**: Single worker design with semaphore-based concurrency control

## Project Structure

```
solar/
├── app/
│   ├── __init__.py          # Package initialization
│   ├── main.py              # FastAPI app initialization
│   └── maps.py              # Main satellite capture & analysis logic
├── copy/
│   └── maps.py              # Backup/older version
├── screenshots/             # Local screenshot storage (optional)
├── Dockerfile               # Production container configuration
├── requirements.txt         # Python dependencies
├── run-dev.bat             # Development startup script (Windows)
└── README.md               # This file
```

## License

[Specify your license here]

## Contributing

[Add contribution guidelines if applicable]

## Support

[Add support/contact information if applicable]

