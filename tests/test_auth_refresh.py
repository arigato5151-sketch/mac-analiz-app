from app.components import auth


def test_current_access_token_refreshes_expired_session(monkeypatch):
    auth.st.session_state.clear()
    auth.st.session_state.update(
        {
            "supabase_access_token": "old",
            "supabase_refresh_token": "refresh",
            "supabase_access_expires_at": 0,
        }
    )

    class Response:
        ok = True

        @staticmethod
        def json():
            return {"access_token": "new", "refresh_token": "rotated", "expires_in": 3600}

    monkeypatch.setattr(auth.requests, "post", lambda *args, **kwargs: Response())

    assert auth.current_access_token() == "new"
    assert auth.st.session_state["supabase_refresh_token"] == "rotated"


def test_refresh_failure_clears_all_session_tokens(monkeypatch):
    auth.st.session_state.clear()
    auth.st.session_state.update(
        {"supabase_access_token": "old", "supabase_refresh_token": "refresh", "supabase_access_expires_at": 0}
    )

    class Response:
        ok = False

        @staticmethod
        def json():
            return {}

    monkeypatch.setattr(auth.requests, "post", lambda *args, **kwargs: Response())

    assert auth.current_access_token() is None
    assert "supabase_refresh_token" not in auth.st.session_state
