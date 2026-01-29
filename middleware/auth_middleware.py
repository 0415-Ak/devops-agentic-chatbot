from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import jwt
from jwt.exceptions import ExpiredSignatureError, InvalidTokenError
import os
from dotenv import load_dotenv

load_dotenv()
JWT_SECRET = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

security = HTTPBearer()
ALLOWED_SOURCES = {"postman", "website", "backend"}

def get_current_user(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> str:
    token = credentials.credentials

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # Extract and validate source header
        source = request.headers.get("X-Client-Source")
        if not source:
            raise HTTPException(status_code=400, detail="Missing X-Client-Source header")

        if source.lower() not in ALLOWED_SOURCES:
            raise HTTPException(
                status_code=403,
                detail=f"Invalid source: '{source}'. Allowed sources: {', '.join(ALLOWED_SOURCES)}"
            )

        return user_id

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")
    
def get_current_user_and_source(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(security)
) -> dict:
    token = credentials.credentials

    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[ALGORITHM])
        user_id = payload.get("sub")

        if not user_id:
            raise HTTPException(status_code=401, detail="Invalid token payload")

        # Extract and validate source header
        source = request.headers.get("X-Client-Source")
        if not source:
            raise HTTPException(status_code=400, detail="Missing X-Client-Source header")

        source = source.lower()
        if source not in ALLOWED_SOURCES:
            raise HTTPException(
                status_code=403,
                detail=f"Invalid source: '{source}'. Allowed sources: {', '.join(ALLOWED_SOURCES)}"
            )

        return {
            "user_id": user_id,
            "source": source
        }

    except ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="Token expired")

    except InvalidTokenError:
        raise HTTPException(status_code=403, detail="Invalid token")
