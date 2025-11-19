from fastapi import FastAPI, File, UploadFile, HTTPException, Depends, Security
from fastapi.responses import JSONResponse
from fastapi.security.api_key import APIKeyHeader
from fastapi.openapi.utils import get_openapi
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import shutil
import os
import uuid


from auth import get_current_user, increment_usage, check_rate_limit, APIKey
from admin import router as admin_router
from parser import (
    extract_text_from_pdf,
    extract_text_from_docx,
    parse_resume_with_gpt,
    clean_ocr_text_with_gpt
)


# --- OpenAPI Security Setup ---
API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)


app = FastAPI(
    title="Resume Parser API",
    description="Professional resume parsing API with authentication, OCR, batch processing, and rate limiting.",
    version="3.0.0"
)


# ----------------------------
# 🚀 GLOBAL CORS FIX
# ----------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://resumifyapi.com",
        "https://www.resumifyapi.com",
        "*"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Custom OpenAPI schema for API key auth in docs
def custom_openapi():
    if app.openapi_schema:
        return app.openapi_schema
    openapi_schema = get_openapi(
        title="Resume Parser API",
        version="3.0.0",
        description="Professional resume parser API with authentication and OCR support",
        routes=app.routes,
    )
    openapi_schema["components"]["securitySchemes"] = {
        "APIKeyHeader": {
            "type": "apiKey",
            "name": API_KEY_NAME,
            "in": "header"
        }
    }
    for path in openapi_schema["paths"]:
        for method in openapi_schema["paths"][path]:
            openapi_schema["paths"][path][method]["security"] = [{"APIKeyHeader": []}]
    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


app.include_router(admin_router)


# ----------------------------
# MAIN ENDPOINT (POST /parse)
# ----------------------------
@app.post("/parse")
async def parse_resume(
    file: UploadFile = File(...),
    current_user: APIKey = Security(get_current_user)
):
    """Parse single resume (PDF/DOCX) with authentication + rate limiting + OCR"""

    check_rate_limit(current_user, 1)

    if not file.filename.lower().endswith(('.pdf', '.docx')):
        raise HTTPException(status_code=400, detail="Only PDF and DOCX files are supported")

    temp_file_path = f"temp_{uuid.uuid4()}_{file.filename}"
    try:
        with open(temp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Extract text from file
        if file.filename.lower().endswith(".pdf"):
            text = extract_text_from_pdf(temp_file_path)
        else:
            text = extract_text_from_docx(temp_file_path)

        # Clean OCR text if needed
        if text and len(text.strip()) > 50:
            text = clean_ocr_text_with_gpt(text)

        # Use GPT to parse all resume data
        parsed_data = parse_resume_with_gpt(text)

        # Increment usage counter
        increment_usage(current_user.key, 1, "parse")

        # Return with "data" wrapper for frontend compatibility
        return {"data": parsed_data}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing file: {str(e)}")

    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.get("/my-usage")
async def get_my_usage(current_user: APIKey = Security(get_current_user)):
    from auth import PLAN_LIMITS
    plan_limits = PLAN_LIMITS[current_user.plan]
    return {
        "user_id": current_user.user_id,
        "plan": current_user.plan,
        "current_usage": current_user.current_usage,
        "monthly_limit": current_user.monthly_limit,
        "usage_percentage": round((current_user.current_usage / current_user.monthly_limit) * 100, 2),
        "remaining": current_user.monthly_limit - current_user.current_usage,
        "plan_features": plan_limits,
        "last_reset": current_user.last_reset
    }


@app.get("/")
async def root():
    return {
        "message": "Resume Parser API is running!",
        "version": "3.0.0",
        "features": [
            "confidence_scoring", "batch_processing",
            "authentication", "rate_limiting", "OCR_support"
        ],
        "get_api_key": "/admin/create-api-key",
        "documentation": "/docs"
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "3.0.0",
        "features": [
            "confidence_scoring", "authentication",
            "rate_limiting", "OCR_support"
        ]
    }
