from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from auth import create_api_key, load_api_keys, save_api_keys, PLAN_LIMITS
from typing import Optional

router = APIRouter(prefix="/admin", tags=["admin"])

class CreateKeyRequest(BaseModel):
    user_id: str
    plan: str = "free"
    email: Optional[str] = None

class APIKeyResponse(BaseModel):
    api_key: str
    user_id: str
    plan: str
    monthly_limit: int
    current_usage: int

@router.post("/create-api-key", response_model=APIKeyResponse)
async def create_new_api_key(request: CreateKeyRequest):
    """Create a new API key for a user"""
    
    if request.plan not in PLAN_LIMITS:
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid plan. Available plans: {list(PLAN_LIMITS.keys())}"
        )
    
    api_key = create_api_key(request.user_id, request.plan)
    api_keys = load_api_keys()
    key_data = api_keys[api_key]
    
    return APIKeyResponse(
        api_key=api_key,
        user_id=key_data.user_id,
        plan=key_data.plan,
        monthly_limit=key_data.monthly_limit,
        current_usage=key_data.current_usage
    )

@router.get("/usage/{api_key}")
async def get_api_key_usage(api_key: str):
    """Get usage statistics for an API key"""
    
    api_keys = load_api_keys()
    if api_key not in api_keys:
        raise HTTPException(status_code=404, detail="API key not found")
    
    key_data = api_keys[api_key]
    plan_limits = PLAN_LIMITS[key_data.plan]
    
    return {
        "user_id": key_data.user_id,
        "plan": key_data.plan,
        "current_usage": key_data.current_usage,
        "monthly_limit": key_data.monthly_limit,
        "usage_percentage": round((key_data.current_usage / key_data.monthly_limit) * 100, 2),
        "remaining": key_data.monthly_limit - key_data.current_usage,
        "plan_features": plan_limits,
        "is_active": key_data.is_active
    }

@router.post("/deactivate/{api_key}")
async def deactivate_api_key(api_key: str):
    """Deactivate an API key"""
    
    api_keys = load_api_keys()
    if api_key not in api_keys:
        raise HTTPException(status_code=404, detail="API key not found")
    
    api_keys[api_key].is_active = False
    save_api_keys(api_keys)
    
    return {"message": "API key deactivated successfully"}

@router.get("/plans")
async def get_pricing_plans():
    """Get available pricing plans"""
    
    return {
        "plans": PLAN_LIMITS,
        "description": {
            "free": "Great for testing and small projects",
            "basic": "Perfect for small businesses",
            "premium": "Ideal for growing companies", 
            "enterprise": "For large-scale operations"
        }
    }
