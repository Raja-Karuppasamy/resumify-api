print("LAUNCHED CODE VERSION: CORS-FIXED")

from fastapi import FastAPI, File, UploadFile, HTTPException, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded
from fastapi.responses import Response
import tempfile
import os
from typing import List

# Import your utilities and modules:
from parser import extract_text_from_pdf, parse_resume_with_gpt
from parser_optimized import parse_resumes_batch
from utils import get_cached_result, cache_result, hash_text, rate_limiter

app = FastAPI(
    title="Resumify API",
    description="AI-Powered Resume Parser with 95.7% accuracy",
    version="1.0.0"
)

# -------------------------------------------------------------------------
# 🚀 CORS (FINAL FIXED VERSION)
# -------------------------------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resumifyapi.com",
        "https://www.resumifyapi.com",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=[
        "Content-Type",
        "X-API-Key",
        "Authorization",
        "Accept",
        "Origin"
    ],
    expose_headers=["X-API-Key"]
)



# -------------------------------------------------------------------------
# Rate Limiting
# -------------------------------------------------------------------------
limiter = Limiter(key_func=get_remote_address)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# API Key
VALID_API_KEY = os.getenv("API_KEY", "demo_key_12345")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key is None:
        raise HTTPException(status_code=401, detail="API key missing. Include X-API-Key header.")
    if x_api_key != VALID_API_KEY:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid API key. Received: {x_api_key}, Expected: {VALID_API_KEY}"
        )
    return x_api_key


# -------------------------------------------------------------------------
# Routes
# -------------------------------------------------------------------------
@app.get("/")
async def root():
    return {
        "message": "Resumify API",
        "version": "1.0.0",
        "docs": "/docs",
        "accuracy": "95.7%",
        "endpoints": {
            "parse_single": "POST /parse",
            "parse_batch": "POST /parse/batch"
        }
    }


@app.post("/parse")
@limiter.limit("100/minute")
async def parse_single(
    request: Request,
    file: UploadFile = File(...),
    x_api_key: str = Header(None)
):
    verify_api_key(x_api_key)

    if not file.filename.endswith('.pdf'):
        raise HTTPException(400, "Only PDF files supported")

    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = extract_text_from_pdf(tmp_path)
        text_hash = hash_text(text)

        # Cache check
        cached = get_cached_result(text_hash)
        if cached:
            return {"status": "success", "data": cached, "cached": True}

        # Rate limiting logic
        rate_limiter.wait_if_needed()

        # Parse using GPT
        result = parse_resume_with_gpt(text)
        cache_result(text_hash, result)

        return {"status": "success", "data": result, "cached": False}

    finally:
        os.unlink(tmp_path)


@app.post("/parse/batch")
@limiter.limit("20/minute")
async def parse_batch(
    request: Request,
    files: List[UploadFile] = File(...),
    x_api_key: str = Header(None)
):
    verify_api_key(x_api_key)

    if len(files) > 5:
        raise HTTPException(400, "Maximum 5 files per batch")

    texts = []
    temp_files = []

    try:
        for file in files:
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp:
                tmp.write(await file.read())
                temp_files.append(tmp.name)
                texts.append(extract_text_from_pdf(tmp.name))

        rate_limiter.wait_if_needed()

        results = parse_resumes_batch(texts)

        for i, result in enumerate(results):
            cache_result(hash_text(texts[i]), result)

        return {"status": "success", "count": len(results), "data": results}

    finally:
        for tmp in temp_files:
            os.unlink(tmp)


@app.get("/health")
async def health():
    return {"status": "healthy"}


@app.get("/debug/check-key")
async def check_key(x_api_key: str = Header(None)):
    return {
        "provided_key": x_api_key,
        "expected_key": VALID_API_KEY,
        "match": x_api_key == VALID_API_KEY
    }
