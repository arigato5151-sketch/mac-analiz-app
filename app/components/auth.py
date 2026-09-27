"""Minimal Supabase Auth session handling for the Streamlit UI."""

from __future__ import annotations

from typing import Any

import requests
import streamlit as st

from config.settings import get_public_supabase_settings
from db.db_client import SupabaseRestClient

try:
    from db.db_client import AuthenticatedSupabaseRestClient
except ImportError:  # Supports a stale Streamlit Cloud build during redeploy.
    class AuthenticatedSupabaseRestClient(SupabaseRestClient):
        """RLS-scoped client fallback for deployments with an older db module."""

        def __init__(self, url: str, anon_key: str, access_token: str, **kwargs: Any) -> None:
            super().__init__(url, anon_key, **kwargs)
            if not access_token:
                raise ValueError("Supabase access token is required")
            self._headers["Authorization"] = f"Bearer {access_token}"


def current_access_token() -> str | None:
    token = st.session_state.get("supabase_access_token")
    return str(token) if token else None


def sign_in(email: str, password: str) -> tuple[bool, str]:
    settings = get_public_supabase_settings()
    try:
        response = requests.post(
            f"{settings.supabase_url}/auth/v1/token?grant_type=password",
            headers={"apikey": settings.supabase_anon_key},
            json={"email": email.strip(), "password": password},
            timeout=settings.request_timeout_seconds,
        )
        if not response.ok:
            return False, "Giriş başarısız. E-posta veya parola hatalı olabilir."
        payload: dict[str, Any] = response.json()
        access_token = str(payload.get("access_token") or "")
        if not access_token:
            return False, "Giriş yanıtı geçersiz."
        st.session_state["supabase_access_token"] = access_token
        st.session_state["supabase_user_email"] = email.strip()
        user = payload.get("user") or {}
        if isinstance(user, dict) and user.get("id"):
            st.session_state["supabase_user_id"] = str(user["id"])
        return True, "Giriş yapıldı."
    except requests.RequestException:
        return False, "Kimlik doğrulama servisine ulaşılamadı."


def sign_out() -> None:
    st.session_state.pop("supabase_access_token", None)
    st.session_state.pop("supabase_user_email", None)
    st.session_state.pop("supabase_user_id", None)


def get_user_db() -> AuthenticatedSupabaseRestClient | None:
    token = current_access_token()
    if not token:
        return None
    settings = get_public_supabase_settings()
    return AuthenticatedSupabaseRestClient(
        settings.supabase_url,
        settings.supabase_anon_key,
        token,
        timeout=settings.request_timeout_seconds,
    )


def current_user_id() -> str | None:
    user_id = st.session_state.get("supabase_user_id")
    return str(user_id) if user_id else None


def render_auth_panel() -> bool:
    token = current_access_token()
    if token:
        email = st.session_state.get("supabase_user_email", "kullanıcı")
        st.caption(f"Oturum açık: {email}")
        if st.button("Çıkış yap", key="supabase_sign_out"):
            sign_out()
            st.rerun()
        return True

    with st.expander("Notları ve takip listesini kalıcılaştır", expanded=False):
        email = st.text_input("E-posta", key="supabase_email")
        password = st.text_input("Parola", type="password", key="supabase_password")
        if st.button("Giriş yap", key="supabase_sign_in"):
            if not email.strip() or not password:
                st.error("E-posta ve parola zorunlu.")
            else:
                success, message = sign_in(email, password)
                (st.success if success else st.error)(message)
                if success:
                    st.rerun()
    return False
