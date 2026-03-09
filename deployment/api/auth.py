"""HTTP Basic Authentication dependency for FastAPI."""
import os
import secrets

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBasic, HTTPBasicCredentials

security = HTTPBasic()

API_USERNAME = os.environ.get("API_USERNAME", "")
API_PASSWORD = os.environ.get("API_PASSWORD", "")


def verify_credentials(credentials: HTTPBasicCredentials = Depends(security)):
    """Validate Basic Auth credentials against environment variables."""
    if not API_USERNAME or not API_PASSWORD:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Authentication not configured on server",
        )
    username_ok = secrets.compare_digest(
        credentials.username.encode("utf-8"),
        API_USERNAME.encode("utf-8"),
    )
    password_ok = secrets.compare_digest(
        credentials.password.encode("utf-8"),
        API_PASSWORD.encode("utf-8"),
    )
    if not username_ok or not password_ok:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid credentials",
            headers={"WWW-Authenticate": "Basic"},
        )
    return credentials.username
