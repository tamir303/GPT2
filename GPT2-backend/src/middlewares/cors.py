from fastapi.middleware.cors import CORSMiddleware

# CORS configuration
cors_config = {
    "allow_origins": ["http://localhost:8000"],  # Adjust as needed
    "allow_credentials": True,
    "allow_methods": ["*"],  # Allow all HTTP methods
    "allow_headers": ["*"],  # Allow all headers
}