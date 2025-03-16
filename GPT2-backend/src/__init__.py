import warnings
from fastapi import FastAPI
from starlette.responses import JSONResponse
from src.etc.logger import CustomLogger, logging
from src.middlewares.auth import OAuth2PasswordBearer, verify_token
from src.middlewares.metrics import handle_metrics, PrometheusMiddleware
from src.middlewares.cors import CORSMiddleware
from src.middlewares.limiter import limiter as FLimiter, RateLimitExceeded, _rate_limit_exceeded_handler

warnings.filterwarnings("ignore")

# Logger setup
logger = CustomLogger(
    log_name='Main',
    log_level=logging.DEBUG,
    log_dir='logs',
    log_filename="runs.log",
).get_logger()

# Initialize FastAPI app
app = FastAPI()

# Rate Limiting Middleware
limiter = FLimiter
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# Authentication Middleware
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://your-frontend-domain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Error Handling Middleware
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(status_code=500, content={"detail": "Internal server error"})

@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    logger.warning(f"ValueError: {exc}")
    return JSONResponse(status_code=400, content={"detail": str(exc)})

# Metrics Middleware
app.add_middleware(PrometheusMiddleware)
app.add_route("/metrics", handle_metrics)
