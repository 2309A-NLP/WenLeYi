"""Small JWT helper used by the Flask API."""
import base64
import hashlib
import hmac
import json
import time
from functools import wraps
from typing import Dict, Optional

from flask import jsonify, request


def _b64url_encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def _b64url_decode(data: str) -> bytes:
    padding = "=" * (-len(data) % 4)
    return base64.urlsafe_b64decode((data + padding).encode("ascii"))


class JWTAuth:
    """Minimal HS256 JWT implementation to avoid an extra hard dependency."""

    def __init__(
        self,
        enabled: bool,
        secret: str,
        expire_hours: int = 24,
        username: str = "admin",
        password: str = "",
    ):
        self.enabled = enabled
        self.secret = secret or "default_secret"
        self.expire_seconds = max(1, int(expire_hours)) * 3600
        self.username = username or "admin"
        self.password = password or ""

    def create_token(self, username: str) -> str:
        now = int(time.time())
        header = {"alg": "HS256", "typ": "JWT"}
        payload = {"sub": username, "iat": now, "exp": now + self.expire_seconds}
        signing_input = ".".join([
            _b64url_encode(json.dumps(header, separators=(",", ":")).encode("utf-8")),
            _b64url_encode(json.dumps(payload, separators=(",", ":")).encode("utf-8")),
        ])
        signature = hmac.new(
            self.secret.encode("utf-8"),
            signing_input.encode("ascii"),
            hashlib.sha256,
        ).digest()
        return f"{signing_input}.{_b64url_encode(signature)}"

    def verify_token(self, token: str) -> Optional[Dict]:
        try:
            header_b64, payload_b64, signature_b64 = token.split(".", 2)
            signing_input = f"{header_b64}.{payload_b64}"
            expected = hmac.new(
                self.secret.encode("utf-8"),
                signing_input.encode("ascii"),
                hashlib.sha256,
            ).digest()
            actual = _b64url_decode(signature_b64)
            if not hmac.compare_digest(expected, actual):
                return None

            payload = json.loads(_b64url_decode(payload_b64).decode("utf-8"))
            if int(payload.get("exp", 0)) < int(time.time()):
                return None
            return payload
        except Exception:
            return None

    def validate_login(self, username: str, password: str) -> bool:
        return username == self.username and password == self.password

    def token_from_request(self) -> str:
        auth_header = request.headers.get("Authorization", "")
        if auth_header.lower().startswith("bearer "):
            return auth_header[7:].strip()
        return (
            request.args.get("token", "").strip()
            or request.args.get("access_token", "").strip()
            or request.cookies.get("access_token", "").strip()
        )

    def require_auth(self, view_func):
        @wraps(view_func)
        def wrapper(*args, **kwargs):
            if not self.enabled:
                return view_func(*args, **kwargs)
            token = self.token_from_request()
            if token and self.verify_token(token):
                return view_func(*args, **kwargs)
            return jsonify({"error": "未授权，请先登录"}), 401

        return wrapper
