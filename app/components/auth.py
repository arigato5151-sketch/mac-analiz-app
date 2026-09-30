"""Minimal Supabase Auth session handling for the Streamlit UI."""

from __future__ import annotations

import time
from typing import Any

import requests
import streamlit as st

from config.settings import get_public_supabase_settings
from db.db_client import AuthenticatedSupabaseRestClient


def current_access_token() -> str | None:
    refresh_session()
    token = st.session_state.get("supabase_access_token")
    return str(token) if token else None


def _store_session(payload: dict[str, Any]) -> bool:
    access_token = str(payload.get("access_token") or "")
    refresh_token = str(payload.get("refresh_token") or "")
    if not access_token or not refresh_token:
        return False
    st.session_state["supabase_access_token"] = access_token
    st.session_state["supabase_refresh_token"] = refresh_token
    expires_in = int(payload.get("expires_in") or 3600)
    st.session_state["supabase_access_expires_at"] = time.time() + max(60, expires_in - 30)
    user = payload.get("user") or {}
    if isinstance(user, dict) and user.get("id"):
        st.session_state["supabase_user_id"] = str(user["id"])
    return True


def refresh_session() -> bool:
    refresh_token = st.session_state.get("supabase_refresh_token")
    expires_at = float(st.session_state.get("supabase_access_expires_at") or 0)
    if not refresh_token or time.time() < expires_at:
        return bool(st.session_state.get("supabase_access_token"))
    settings = get_public_supabase_settings()
    try:
        response = requests.post(
            f"{settings.supabase_url}/auth/v1/token?grant_type=refresh_token",
            headers={"apikey": settings.supabase_anon_key},
            json={"refresh_token": str(refresh_token)},
            timeout=settings.request_timeout_seconds,
        )
        if not response.ok or not _store_session(response.json()):
            sign_out()
            return False
        return True
    except (requests.RequestException, ValueError, TypeError):
        sign_out()
        return False


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
        if not _store_session(payload):
            return False, "GiriÅŸ yanÄ±tÄ± geÃ§ersiz."
        st.session_state["supabase_user_email"] = email.strip()
        return True, "Giriş yapıldı."
    except requests.RequestException:
        return False, "Kimlik doğrulama servisine ulaşılamadı."


def sign_out() -> None:
    st.session_state.pop("supabase_access_token", None)
    st.session_state.pop("supabase_refresh_token", None)
    st.session_state.pop("supabase_access_expires_at", None)
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
