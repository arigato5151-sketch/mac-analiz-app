from __future__ import annotations

from notifications.daily_summary import build_morning_message, build_night_message
from notifications.telegram import TelegramError, send_telegram_message


class FakeResponse:
    def __init__(self, *, ok: bool, status_code: int, payload: dict[str, object]) -> None:
        self.ok = ok
        self.status_code = status_code
        self._payload = payload

    def json(self) -> dict[str, object]:
        return self._payload


class FakeSession:
    def __init__(self, response: FakeResponse) -> None:
        self.response = response
        self.url = ""
        self.payload: dict[str, object] = {}

    def post(self, url: str, **kwargs: object) -> FakeResponse:
        self.url = url
        self.payload = kwargs["json"]  # type: ignore[assignment,index]
        return self.response


def test_telegram_send_posts_expected_payload_without_returning_token() -> None:
    session = FakeSession(FakeResponse(ok=True, status_code=200, payload={"ok": True}))

    send_telegram_message("Merhaba", bot_token="secret-token", chat_id="42", session=session)

    assert session.url.endswith("/botsecret-token/sendMessage")
    assert session.payload["chat_id"] == "42"


def test_telegram_error_does_not_include_credentials() -> None:
    session = FakeSession(FakeResponse(ok=False, status_code=401, payload={}))

    try:
        send_telegram_message("Merhaba", bot_token="secret-token", chat_id="42", session=session)
    except TelegramError as exc:
        assert "secret-token" not in str(exc)
    else:
        raise AssertionError("TelegramError expected")


def test_morning_and_night_messages_include_useful_summary() -> None:
    morning = build_morning_message(
        [{"id": 1, "league_id": 39, "home_team_id": 10, "away_team_id": 20, "match_date": "2026-08-30T16:00:00+00:00"}],
        [{"match_id": 1, "prob_home_win": 0.62, "prob_draw": 0.20, "prob_away_win": 0.18, "prob_over_2_5": 0.55, "prob_btts": 0.45, "predicted_at": "2026-08-29T10:00:00+00:00"}],
        [{"id": 10, "name": "Ev"}, {"id": 20, "name": "Deplasman"}],
        [{"id": 39, "name": "Premier League"}],
    )
    night = build_night_message([{"was_correct": True, "brier_score": 0.2}])

    assert "Ev — Deplasman" in morning
    assert "Ev kazanır (%62)" in morning
    assert "%100.0" in night
