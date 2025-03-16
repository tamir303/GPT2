from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from src.middlewares.auth import OAuth2PasswordBearer
from src.middlewares.metrics import handle_metrics, PrometheusMiddleware
from src.middlewares.limiter import limiter as FLimiter, RateLimitExceeded, _rate_limit_exceeded_handler

def setup_middlewares(app: FastAPI):
    """Mount all middlewares to the FastAPI app"""

    # Rate Limiting Middleware
    app.state.limiter = FLimiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

    # Authentication Middleware
    app.dependency_overrides[OAuth2PasswordBearer] = OAuth2PasswordBearer(tokenUrl="token")

    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["https://your-frontend-domain.com"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Metrics Middleware
    app.add_middleware(PrometheusMiddleware)
    app.add_route("/metrics", handle_metrics)

