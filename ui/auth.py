"""Dashboard accounts, credential checks, and signed browser sessions."""

from __future__ import annotations

import argparse
import base64
import getpass
import hashlib
import hmac
import json
import os
import re
import secrets
import time
from pathlib import Path

from sqlalchemy import create_engine, text
from uk_training_data_prep.database import normalize_database_url


ROLES = {"admin", "super_admin"}
EMAIL_RE = re.compile(r"^[^\s@]+@[^\s@]+\.[^\s@]+$")
SESSION_SECONDS = 8 * 60 * 60
GATE_SECONDS = 365 * 24 * 60 * 60
_engine = None
_engine_url = None


def database_url() -> str:
    configured = os.environ.get("DASHBOARD_AUTH_DATABASE_URL") or os.environ.get("DATABASE_URL")
    if configured:
        return normalize_database_url(configured)
    path = Path(os.environ.get("DASHBOARD_AUTH_DB_PATH", "data/dashboard_auth.sqlite3")).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{path.as_posix()}"


def engine():
    global _engine, _engine_url
    url = database_url()
    if _engine is None or url != _engine_url:
        _engine = create_engine(url, pool_pre_ping=True)
        _engine_url = url
    return _engine


def init_db() -> None:
    with engine().begin() as connection:
        connection.execute(text("""CREATE TABLE IF NOT EXISTS dashboard_users (
            id VARCHAR(32) PRIMARY KEY,
            email VARCHAR(320) NOT NULL UNIQUE,
            role VARCHAR(20) NOT NULL,
            password_hash VARCHAR(256),
            google_sub VARCHAR(255) UNIQUE,
            active BOOLEAN NOT NULL DEFAULT TRUE,
            token_version INTEGER NOT NULL DEFAULT 0
        )"""))


def normalize_email(email: str) -> str:
    email = email.strip().lower()
    if len(email) > 320 or not EMAIL_RE.fullmatch(email):
        raise ValueError("Enter a valid email address.")
    return email


def hash_password(password: str) -> str:
    if len(password) < 12 or len(password) > 1024:
        raise ValueError("Password must contain at least 12 characters.")
    salt = secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310000)
    return f"pbkdf2_sha256$310000${salt.hex()}${digest.hex()}"


def check_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        name, rounds, salt, digest = stored.split("$")
        if name != "pbkdf2_sha256":
            return False
        candidate = hashlib.pbkdf2_hmac("sha256", password.encode(), bytes.fromhex(salt), int(rounds))
        return hmac.compare_digest(candidate, bytes.fromhex(digest))
    except (ValueError, TypeError):
        return False


def get_user(email: str):
    with engine().connect() as connection:
        row = connection.execute(text("SELECT * FROM dashboard_users WHERE email=:email"), {"email": email}).mappings().first()
        return dict(row) if row else None


def get_user_by_id(user_id: str):
    with engine().connect() as connection:
        row = connection.execute(text("SELECT * FROM dashboard_users WHERE id=:id"), {"id": user_id}).mappings().first()
        return dict(row) if row else None


def list_users() -> list[dict]:
    with engine().connect() as connection:
        return [{"id": row["id"], "email": row["email"], "role": row["role"], "active": bool(row["active"]), "has_password": bool(row["password_hash"]), "google_connected": bool(row["google_sub"])} for row in connection.execute(text("SELECT * FROM dashboard_users ORDER BY email")).mappings()]


def service_login_allowed(role: str) -> bool:
    """Shared tokens can bootstrap accounts, but cannot bypass individual revocation."""
    with engine().connect() as connection:
        if role == "super_admin":
            count = connection.execute(text("SELECT COUNT(*) FROM dashboard_users WHERE role='super_admin'")).scalar_one()
        elif role == "admin":
            count = connection.execute(text("SELECT COUNT(*) FROM dashboard_users")).scalar_one()
        else:
            return False
    return count == 0


def service_token(role: str) -> str:
    if role == "super_admin":
        return os.environ.get("DASHBOARD_LOGIN_TOKEN", "").strip() or os.environ.get("DASHBOARD_SUPER_ADMIN_TOKEN", "").strip()
    if role == "admin":
        return os.environ.get("DASHBOARD_ADMIN_TOKEN", "").strip()
    return ""


def create_user(email: str, role: str, password: str | None = None) -> dict:
    email = normalize_email(email)
    if role not in ROLES:
        raise ValueError("Invalid role.")
    if not password and not os.environ.get("GOOGLE_CLIENT_ID", "").strip():
        raise ValueError("Set a password or configure Google sign in.")
    hashed = hash_password(password) if password else None
    user_id = secrets.token_hex(16)
    with engine().begin() as connection:
        connection.execute(text("INSERT INTO dashboard_users (id,email,role,password_hash,active,token_version) VALUES (:id,:email,:role,:password,TRUE,0)"), {"id": user_id, "email": email, "role": role, "password": hashed})
    return get_user_by_id(user_id)


def set_user_access(user_id: str, active: bool) -> bool:
    with engine().begin() as connection:
        result = connection.execute(text("UPDATE dashboard_users SET active=:active, token_version=token_version+1 WHERE id=:id"), {"active": active, "id": user_id})
        return result.rowcount == 1


def login_password(email: str, password: str):
    try:
        user = get_user(normalize_email(email))
    except ValueError:
        return None
    return user if user and user["active"] and check_password(password, user["password_hash"]) else None


def login_google(credential: str):
    client_id = os.environ.get("GOOGLE_CLIENT_ID", "").strip()
    if not client_id:
        raise ValueError("Google sign in is not configured.")
    from google.auth.transport import requests
    from google.oauth2 import id_token
    try:
        claims = id_token.verify_oauth2_token(credential, requests.Request(), client_id)
    except ValueError:
        return None
    if claims.get("email_verified") is not True or not claims.get("sub"):
        return None
    user = get_user(normalize_email(claims["email"]))
    if not user or not user["active"]:
        return None
    with engine().begin() as connection:
        if user["google_sub"] and user["google_sub"] != claims["sub"]:
            return None
        if not user["google_sub"]:
            connection.execute(text("UPDATE dashboard_users SET google_sub=:sub WHERE id=:id AND google_sub IS NULL"), {"sub": claims["sub"], "id": user["id"]})
    return get_user_by_id(user["id"])


def _encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _decode(value: str) -> bytes:
    return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))


def session_secret() -> bytes:
    override = os.environ.get("DASHBOARD_SESSION_SECRET", "").strip()
    if override:
        if len(override) < 32:
            raise RuntimeError("DASHBOARD_SESSION_SECRET must contain at least 32 characters.")
        return override.encode()

    configured = [
        ("login", os.environ.get("DASHBOARD_LOGIN_TOKEN", "").strip()),
        ("super_admin", os.environ.get("DASHBOARD_SUPER_ADMIN_TOKEN", "").strip()),
        ("admin", os.environ.get("DASHBOARD_ADMIN_TOKEN", "").strip()),
    ]
    material = b"\0".join(f"{name}:{value}".encode() for name, value in configured if value)
    if not material:
        raise RuntimeError(
            "Dashboard authentication is not configured. Set DASHBOARD_LOGIN_TOKEN "
            "(or DASHBOARD_ADMIN_TOKEN/DASHBOARD_SUPER_ADMIN_TOKEN)."
        )
    return hashlib.sha256(b"uk-smart-grid-dashboard/session-signing-key/v1\0" + material).digest()


def issue_session(user: dict | None = None, service_role: str | None = None, ttl: int = SESSION_SECONDS, purpose: str = "session") -> str:
    now = int(time.time())
    if purpose not in {"session", "gate"}:
        raise ValueError("Invalid token purpose.")
    payload = {"iat": now, "exp": now + ttl, "purpose": purpose}
    if user:
        payload.update({"sub": user["id"], "version": user["token_version"]})
    elif service_role in ROLES:
        if purpose == "session" and not service_login_allowed(service_role):
            raise ValueError("Sign in with your account after entering the access token.")
        secret = service_token(service_role)
        if not secret:
            raise ValueError("Token login is not configured.")
        payload.update({"service": service_role, "fingerprint": hashlib.sha256(secret.encode()).hexdigest()})
    else:
        raise ValueError("Missing session identity.")
    header = _encode(b'{"alg":"HS256","typ":"JWT"}')
    body = _encode(json.dumps(payload, separators=(",", ":")).encode())
    signing_input = f"{header}.{body}".encode()
    signature = _encode(hmac.new(session_secret(), signing_input, hashlib.sha256).digest())
    return f"{header}.{body}.{signature}"


def session_identity(token: str, purpose: str = "session"):
    try:
        header, body, signature = token.split(".")
        if header != _encode(b'{"alg":"HS256","typ":"JWT"}'):
            return None
        expected = hmac.new(session_secret(), f"{header}.{body}".encode(), hashlib.sha256).digest()
        if not hmac.compare_digest(expected, _decode(signature)):
            return None
        claims = json.loads(_decode(body))
        now = int(time.time())
        if claims.get("purpose") != purpose or claims.get("exp", 0) <= now or claims.get("iat", 0) > now + 60:
            return None
        if "sub" in claims:
            user = get_user_by_id(claims["sub"])
            return user if user and user["active"] and user["token_version"] == claims.get("version") else None
        role = claims.get("service")
        if role in ROLES:
            secret = service_token(role)
            if secret and (purpose == "gate" or service_login_allowed(role)) and hmac.compare_digest(hashlib.sha256(secret.encode()).hexdigest(), claims.get("fingerprint", "")):
                return {"id": None, "email": "Token access", "role": role, "active": True}
    except (ValueError, TypeError, KeyError, json.JSONDecodeError):
        pass
    return None


def login_token(value: str):
    if not value:
        return None
    expected = os.environ.get("DASHBOARD_LOGIN_TOKEN", "")
    if expected and hmac.compare_digest(value, expected):
        return "super_admin"
    for role, name in (("super_admin", "DASHBOARD_SUPER_ADMIN_TOKEN"), ("admin", "DASHBOARD_ADMIN_TOKEN")):
        expected = os.environ.get(name, "")
        if expected and hmac.compare_digest(value, expected):
            return role
    return None


def bootstrap_super_admin(email: str, password: str) -> dict:
    init_db()
    with engine().connect() as connection:
        count = connection.execute(text("SELECT COUNT(*) FROM dashboard_users WHERE role='super_admin' AND active=TRUE")).scalar_one()
    if count:
        raise ValueError("A super admin already exists. Create further accounts from the super admin page.")
    return create_user(email, "super_admin", password)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bootstrap the first dashboard super admin")
    parser.add_argument("command", choices=["create-super-admin"])
    parser.add_argument("--email", required=True)
    args = parser.parse_args()
    password = getpass.getpass("Password (at least 12 characters): ")
    confirmation = getpass.getpass("Confirm password: ")
    if password != confirmation:
        parser.error("Passwords do not match.")
    user = bootstrap_super_admin(args.email, password)
    print(f"Created super admin {user['email']}.")


if __name__ == "__main__":
    main()
