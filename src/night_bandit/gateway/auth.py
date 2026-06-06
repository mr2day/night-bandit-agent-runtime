"""IdP (OIDC) JWT verification for the WS gateway.

In production, the first WS frame carries a raw JWT which we verify against
the IdP's JWKS (RS256, issuer + audience checked); the user id is the
token's ``sub``. With ``NB_DEV_BYPASS_AUTH=true`` we skip verification for
localhost smoke tests and derive a stable dev user id from the token (or a
constant when none is supplied).
"""

from __future__ import annotations

import hashlib

import jwt
from jwt import PyJWKClient

from ..config import get_settings

_jwks_client: PyJWKClient | None = None


class AuthError(Exception):
    pass


def _client() -> PyJWKClient:
    global _jwks_client
    if _jwks_client is None:
        _jwks_client = PyJWKClient(get_settings().idp_jwks_uri)
    return _jwks_client


def verify_token(token: str | None) -> str:
    """Return the authenticated user id, or raise AuthError."""
    settings = get_settings()

    if settings.dev_bypass_auth:
        if not token:
            return "dev-user"
        # Derive a stable id from the token so distinct dev tokens map to
        # distinct users without verifying anything.
        return "dev-" + hashlib.sha256(token.encode()).hexdigest()[:12]

    if not token:
        raise AuthError("missing auth token")
    try:
        signing_key = _client().get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256"],
            audience=settings.idp_audience,
            issuer=settings.idp_issuer,
        )
    except Exception as e:  # noqa: BLE001 - surface any verification failure uniformly
        raise AuthError(f"token verification failed: {e}") from e
    sub = claims.get("sub")
    if not sub:
        raise AuthError("token has no sub claim")
    return str(sub)
