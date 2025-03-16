import sys
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from dotenv import load_dotenv
import os
import jwt

load_dotenv()

# JWT configuration (store SECRET_KEY securely, e.g., in environment variables)
SECRET_KEY = os.getenv("API_KEY")
ALGORITHM: str = os.getenv("ALGORITHM", "HS256")

if not SECRET_KEY:
    print("ERROR: API_KEY is not set in environment variables.", file=sys.stderr)
    sys.exit(1)  # Stop execution if API_KEY is missing

# OAuth2 scheme for token extraction
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Token verification function
def verify_token(token: str = Depends(oauth2_scheme)):
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid token",
            headers={"WWW-Authenticate": "Bearer"},
        )