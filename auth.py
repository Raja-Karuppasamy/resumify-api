import hashlib
import secrets
from datetime import datetime, timedelta
from typing import Optional, Dict
from fastapi import HTTPException, Header, Depends
from pydantic import BaseModel
import json
import os

# Simple file-based storage for API keys (in production, use a database)
API_KEYS_FILE = "api_keys.json"
USAGE_LOGS_FILE = "usage_logs.json"

class APIKey(BaseModel):
    key: str
    user_id: str
    plan: str  # "free", "basic", "premium", "enterprise"
    created_at: datetime
    is_active: bool = True
    monthly_limit: int
    current_usage: int = 0
    last_reset: datetime

class UsageLog(BaseModel):
    api_key: str
    endpoint: str
    timestamp: datetime
    files_processed: int = 1

# Rate limits per plan
PLAN_LIMITS = {
    "free": {
        "monthly_limit": 100,
        "rate_per_minute": 5,
        "max_batch_size": 5
    },
    "basic": {
        "monthly_limit": 1000,
        "rate_per_minute": 20,
        "max_batch_size": 20
    },
    "premium": {
        "monthly_limit": 10000,
        "rate_per_minute": 50,
        "max_batch_size": 50
    },
    "enterprise": {
        "monthly_limit": 100000,
        "rate_per_minute": 200,
        "max_batch_size": 100
    }
}

def load_api_keys() -> Dict[str, APIKey]:
    """Load API keys from file"""
    if not os.path.exists(API_KEYS_FILE):
        return {}
    
    try:
        with open(API_KEYS_FILE, 'r') as f:
            data = json.load(f)
            return {k: APIKey(**v) for k, v in data.items()}
    except:
        return {}

def save_api_keys(api_keys: Dict[str, APIKey]):
    """Save API keys to file"""
    data = {k: v.dict() for k, v in api_keys.items()}
    # Convert datetime objects to strings for JSON serialization
    for key_data in data.values():
        key_data['created_at'] = key_data['created_at'].isoformat()
        key_data['last_reset'] = key_data['last_reset'].isoformat()
    
    with open(API_KEYS_FILE, 'w') as f:
        json.dump(data, f, indent=2)

def load_usage_logs() -> list:
    """Load usage logs from file"""
    if not os.path.exists(USAGE_LOGS_FILE):
        return []
    
    try:
        with open(USAGE_LOGS_FILE, 'r') as f:
            return json.load(f)
    except:
        return []

def save_usage_log(log_entry: UsageLog):
    """Save usage log entry"""
    logs = load_usage_logs()
    log_dict = log_entry.dict()
    log_dict['timestamp'] = log_dict['timestamp'].isoformat()
    logs.append(log_dict)
    
    # Keep only last 10000 entries
    if len(logs) > 10000:
        logs = logs[-10000:]
    
    with open(USAGE_LOGS_FILE, 'w') as f:
        json.dump(logs, f, indent=2)

def generate_api_key() -> str:
    """Generate a secure API key"""
    return f"rp_{secrets.token_urlsafe(32)}"

def create_api_key(user_id: str, plan: str = "free") -> str:
    """Create a new API key for a user"""
    api_keys = load_api_keys()
    
    # Check if user already has an active key
    for key_data in api_keys.values():
        if key_data.user_id == user_id and key_data.is_active:
            return key_data.key
    
    # Generate new key
    api_key = generate_api_key()
    now = datetime.now()
    
    api_keys[api_key] = APIKey(
        key=api_key,
        user_id=user_id,
        plan=plan,
        created_at=now,
        monthly_limit=PLAN_LIMITS[plan]["monthly_limit"],
        last_reset=now
    )
    
    save_api_keys(api_keys)
    return api_key

def verify_api_key(api_key: str) -> APIKey:
    """Verify and return API key data"""
    api_keys = load_api_keys()
    
    if api_key not in api_keys:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    key_data = api_keys[api_key]
    
    if not key_data.is_active:
        raise HTTPException(status_code=401, detail="API key is disabled")
    
    # Check if we need to reset monthly usage
    now = datetime.now()
    if now.month != key_data.last_reset.month or now.year != key_data.last_reset.year:
        key_data.current_usage = 0
        key_data.last_reset = now
        api_keys[api_key] = key_data
        save_api_keys(api_keys)
    
    # Check monthly limit
    if key_data.current_usage >= key_data.monthly_limit:
        raise HTTPException(
            status_code=429, 
            detail=f"Monthly limit of {key_data.monthly_limit} requests exceeded"
        )
    
    return key_data

def increment_usage(api_key: str, files_processed: int = 1, endpoint: str = "parse-resume"):
    """Increment usage count for API key"""
    api_keys = load_api_keys()
    
    if api_key in api_keys:
        api_keys[api_key].current_usage += files_processed
        save_api_keys(api_keys)
        
        # Log usage
        log_entry = UsageLog(
            api_key=api_key,
            endpoint=endpoint,
            timestamp=datetime.now(),
            files_processed=files_processed
        )
        save_usage_log(log_entry)

def get_current_user(x_api_key: Optional[str] = Header(None)) -> APIKey:
    """Dependency to get current authenticated user"""
    if not x_api_key:
        raise HTTPException(
            status_code=401,
            detail="API key required. Include 'X-API-Key' header."
        )
    
    return verify_api_key(x_api_key)

def check_rate_limit(api_key_data: APIKey, files_count: int = 1):
    """Check if request exceeds rate limits"""
    plan_limits = PLAN_LIMITS[api_key_data.plan]
    
    # Check batch size limit
    if files_count > plan_limits["max_batch_size"]:
        raise HTTPException(
            status_code=429,
            detail=f"Batch size {files_count} exceeds limit of {plan_limits['max_batch_size']} for {api_key_data.plan} plan"
        )
    
    # TODO: Implement per-minute rate limiting (requires Redis/memory store in production)
    # For now, we'll just check monthly limits which is handled in verify_api_key
