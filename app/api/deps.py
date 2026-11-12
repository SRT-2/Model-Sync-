from __future__ import annotations

import secrets
from typing import Optional
from fastapi import Header, HTTPException, status

from app.config import get_settings


async def verify_api_key(
    authorization: Optional[str] = Header(None, alias="Authorization"),
) -> None:
    """Validate Bearer token if ROUTER_API_KEY is configured."""
    settings = get_settings()
    configured_key = settings.router_api_key

    # If no router_api_key is set, auth is disabled
    if not configured_key or not configured_key.strip():
        return

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )

    parts = authorization.split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header format. Expected 'Bearer <token>'",
            headers={"WWW-Authenticate": "Bearer"},
        )

    provided_token = parts[1]
    if not secrets.compare_digest(provided_token, configured_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"},
        )
