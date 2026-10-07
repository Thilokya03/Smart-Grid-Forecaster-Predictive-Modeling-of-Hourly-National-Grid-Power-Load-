"""Credential, session, and role enforcement checks for the dashboard."""

import json
import pytest
import sys
import threading
import types
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ui import auth
from ui.pipeline_dashboard import DashboardHandler, required_role_for_page


def test_accounts_sessions_and_revocation(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_DB_PATH", str(tmp_path / "accounts.sqlite3"))
    monkeypatch.delenv("DASHBOARD_AUTH_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
    monkeypatch.setenv("DASHBOARD_LOGIN_TOKEN", "test-deployment-login-token")
    monkeypatch.setenv("DASHBOARD_SUPER_ADMIN_TOKEN", "test-super-admin-token")
    auth.init_db()

    token_session = auth.issue_session(service_role="super_admin")
    gate_token = auth.issue_session(service_role="super_admin", ttl=900, purpose="gate")
    assert auth.session_identity(token_session)["role"] == "super_admin"
    assert auth.session_identity(gate_token) is None
    assert auth.session_identity(gate_token, "gate")["role"] == "super_admin"

    super_user = auth.bootstrap_super_admin("Owner@Example.com", "a secure super password")
    assert super_user["email"] == "owner@example.com"
    assert auth.session_identity(token_session) is None
    assert auth.session_identity(gate_token, "gate")["role"] == "super_admin"
    assert not auth.service_login_allowed("super_admin")
    admin = auth.create_user("admin@example.com", "admin", "a secure admin password")
    assert auth.login_password("admin@example.com", "wrong") is None
    assert auth.login_password("admin@example.com", "a secure admin password")["id"] == admin["id"]
    assert auth.login_token("wrong") is None
    assert auth.login_token("test-super-admin-token") == "super_admin"
    monkeypatch.delenv("DASHBOARD_SUPER_ADMIN_TOKEN")
    assert auth.session_identity(token_session) is None
    assert auth.session_identity(gate_token, "gate") is None

    session = auth.issue_session(admin)
    assert auth.session_identity(session)["role"] == "admin"
    assert auth.session_identity(session + "x") is None
    assert auth.set_user_access(admin["id"], False)
    assert auth.session_identity(session) is None
    assert auth.login_password("admin@example.com", "a secure admin password") is None
    assert auth.set_user_access(admin["id"], True)
    assert auth.session_identity(session) is None  # old session remains revoked


def test_google_login_requires_verified_existing_account(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_DB_PATH", str(tmp_path / "google.sqlite3"))
    monkeypatch.delenv("DASHBOARD_AUTH_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setenv("GOOGLE_CLIENT_ID", "test-client-id")
    auth.init_db()
    admin = auth.create_user("admin@example.com", "admin")
    claims = {"email": "admin@example.com", "email_verified": True, "sub": "google-user-1"}
    google = types.ModuleType("google")
    google.__path__ = []
    google_auth = types.ModuleType("google.auth")
    google_auth.__path__ = []
    transport = types.ModuleType("google.auth.transport")
    transport.requests = types.SimpleNamespace(Request=lambda: object())
    oauth2 = types.ModuleType("google.oauth2")
    oauth2.__path__ = []
    oauth2.id_token = types.SimpleNamespace(verify_oauth2_token=lambda token, request, audience: dict(claims))
    for name, module in (("google", google), ("google.auth", google_auth), ("google.auth.transport", transport), ("google.oauth2", oauth2)):
        monkeypatch.setitem(sys.modules, name, module)
    assert auth.login_google("credential")["id"] == admin["id"]
    claims["email_verified"] = False
    assert auth.login_google("credential") is None
    claims["email_verified"] = True
    claims["sub"] = "different-google-user"
    assert auth.login_google("credential") is None
    claims["sub"] = "google-user-1"
    claims["email"] = "unknown@example.com"
    assert auth.login_google("credential") is None


def test_session_secret_is_derived_without_session_environment(monkeypatch):
    monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
    monkeypatch.setenv("DASHBOARD_LOGIN_TOKEN", "deployment-token")
    first = auth.session_secret()
    assert len(first) == 32
    assert first == auth.session_secret()
    monkeypatch.delenv("DASHBOARD_LOGIN_TOKEN")
    monkeypatch.delenv("DASHBOARD_ADMIN_TOKEN", raising=False)
    monkeypatch.delenv("DASHBOARD_SUPER_ADMIN_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match="authentication is not configured"):
        auth.session_secret()


def test_http_login_roles_and_account_management(tmp_path, monkeypatch):
    monkeypatch.setenv("DASHBOARD_AUTH_DB_PATH", str(tmp_path / "http.sqlite3"))
    monkeypatch.delenv("DASHBOARD_AUTH_DATABASE_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.delenv("DASHBOARD_SESSION_SECRET", raising=False)
    monkeypatch.setenv("DASHBOARD_LOGIN_TOKEN", "test-deployment-login-token")
    monkeypatch.setenv("DASHBOARD_ADMIN_TOKEN", "test-admin-token")
    monkeypatch.setenv("DASHBOARD_SUPER_ADMIN_TOKEN", "test-super-token")
    auth.init_db()
    auth.bootstrap_super_admin("owner@example.com", "a secure super password")
    server = ThreadingHTTPServer(("127.0.0.1", 0), DashboardHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{server.server_port}"

    def request(path, body=None, cookie=None):
        headers = {"Content-Type": "application/json"}
        if cookie:
            headers["Cookie"] = cookie
        req = Request(base + path, data=json.dumps(body).encode() if body is not None else None, headers=headers)
        try:
            return urlopen(req)
        except HTTPError as exc:
            return exc

    try:
        assert request("/api/admin/users").status == 403
        assert request("/api/admin/users?token=anything").status == 403
        assert b"Enter your access token" in request("/login").read()
        assert request("/static/login.html").status == 404
        assert request("/api/auth/unlock", {"token": "wrong"}).status == 401
        login_gate_response = request("/api/auth/unlock", {"token": "test-deployment-login-token"})
        assert login_gate_response.status == 200
        login_gate_cookie = login_gate_response.headers["Set-Cookie"].split(";", 1)[0]
        assert b"Welcome back" in request("/login", cookie=login_gate_cookie).read()
        assert request("/api/auth/login", {"method": "password", "email": "owner@example.com", "password": "a secure super password"}).status == 403
        gate_response = request("/api/auth/unlock", {"token": "test-super-token"})
        gate_cookie = gate_response.headers["Set-Cookie"].split(";", 1)[0]
        assert b"Welcome back" in request("/login", cookie=gate_cookie).read()
        admin_gate_response = request("/api/auth/unlock", {"token": "test-admin-token"})
        admin_gate_cookie = admin_gate_response.headers["Set-Cookie"].split(";", 1)[0]
        assert request("/api/auth/login", {"email": "owner@example.com", "password": "a secure super password"}, admin_gate_cookie).status == 401
        assert request("/api/auth/login", {"method": "token", "token": "test-super-token"}).status == 403
        assert request("/api/auth/login", {"method": "password", "email": "owner@example.com", "password": "bad"}, gate_cookie).status == 401
        response = request("/api/auth/login", {"method": "password", "email": "owner@example.com", "password": "a secure super password"}, gate_cookie)
        assert response.status == 200
        cookie = response.headers["Set-Cookie"].split(";", 1)[0]
        assert "HttpOnly" in response.headers["Set-Cookie"]
        assert request("/api/admin/users", cookie=cookie).status == 200
        response = request("/api/admin/users", {"email": "admin@example.com", "role": "admin", "password": "a secure admin password"}, cookie)
        assert response.status == 201
        admin_id = json.load(response)["id"]
        response = request("/api/admin/users", {"email": "second@example.com", "role": "super_admin", "password": "another secure password"}, cookie)
        assert response.status == 201
        second_id = json.load(response)["id"]
        second_login = request("/api/auth/login", {"email": "second@example.com", "password": "another secure password"}, gate_cookie)
        assert json.load(second_login)["role"] == "super_admin"
        assert request(f"/api/admin/users/{second_id}/access", {"active": False}, cookie).status == 400
        admin_response = request("/api/auth/login", {"email": "admin@example.com", "password": "a secure admin password"}, gate_cookie)
        admin_cookie = admin_response.headers["Set-Cookie"].split(";", 1)[0]
        assert request("/api/admin/users", cookie=admin_cookie).status == 403
        logout = request("/api/auth/logout", {}, cookie)
        assert logout.status == 200
        assert all("dashboard_gate" not in value for value in logout.headers.get_all("Set-Cookie"))
        assert b"Welcome back" in request("/login", cookie=gate_cookie).read()
        assert b"Create a new admin" in request("/super-admin/create-admin", cookie=cookie).read()
        assert required_role_for_page("/super-admin/create-admin") == "super_admin"
        assert request("/static/admin_accounts.html", cookie=cookie).status == 404
        assert request("/api/admin/users").status == 403
        assert request("/api/pipeline-health", cookie=admin_cookie).status == 403
        assert request(f"/api/admin/users/{admin_id}/access", {"active": False}, cookie).status == 200
        assert request("/api/auth/me", cookie=admin_cookie).read() == b'{"email": null, "role": "public"}'
        assert request("/api/admin/users", cookie=admin_cookie).status == 403
    finally:
        server.shutdown()
        server.server_close()
