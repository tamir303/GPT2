from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Initialize Limiter using client IP as the key
limiter = Limiter(key_func=get_remote_address)