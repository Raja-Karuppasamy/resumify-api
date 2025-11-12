import redis
import json
import hashlib
import time
from threading import Lock
import os

# Redis client (will use Railway's Redis URL)
redis_url = os.getenv('REDIS_URL', 'redis://localhost:6379/0')
redis_client = redis.from_url(redis_url, decode_responses=True)

def hash_text(text: str) -> str:
    return hashlib.md5(text.encode()).hexdigest()

def get_cached_result(text_hash: str):
    try:
        cached = redis_client.get(f"resume:{text_hash}")
        if cached:
            return json.loads(cached)
    except:
        pass
    return None

def cache_result(text_hash: str, result: dict, ttl=86400):
    try:
        redis_client.setex(f"resume:{text_hash}", ttl, json.dumps(result))
    except:
        pass

class RateLimiter:
    def __init__(self, max_calls_per_minute=450):
        self.max_calls = max_calls_per_minute
        self.calls = []
        self.lock = Lock()
    
    def wait_if_needed(self):
        with self.lock:
            now = time.time()
            self.calls = [t for t in self.calls if now - t < 60]
            
            if len(self.calls) >= self.max_calls:
                sleep_time = 60 - (now - self.calls[0])
                if sleep_time > 0:
                    time.sleep(sleep_time)
                self.calls = []
            
            self.calls.append(now)

rate_limiter = RateLimiter()
