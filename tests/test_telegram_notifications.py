from __future__ import annotations

from notifications.daily_summary import build_morning_messages, build_night_message
from notifications.telegram import TelegramError, send_many_from_environment, send_telegram_message


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


def test_morning_message_contains_all_prediction_markets_for_one_fixture() -> None:
    messages = build_morning_messages(
        [{"id": 1, "league_id": 39, "home_team_id": 10, "away_team_id": 20, "match_date": "2026-08-30T16:00:00+00:00"}],
        [{"match_id": 1, "prob_home_win": 0.62, "prob_draw": 0.20, "prob_away_win": 0.18, "prob_over_2_5": 0.55, "prob_btts": 0.45, "predicted_at": "2026-08-29T10:00:00+00:00"}],
        [{"id": 10, "name": "Ev"}, {"id": 20, "name": "Deplasman"}],
        [{"id": 39, "name": "Premier League"}],
    )
    assert len(messages) == 1
    assert "Ev — Deplasman" in messages[0]
    assert "1-X-2: 1 %62 · X %20 · 2 %18" in messages[0]
    assert "Üst/Alt 2.5: Üst %55 · Alt %45" in messages[0]
    assert "KG Var/Yok: Var %45 · Yok %55" in messages[0]


def test_night_message_includes_recent_performance() -> None:
    night = build_night_message([{"was_correct": True, "brier_score": 0.2}])

    assert "%100.0" in night


def test_send_many_skips_when_credentials_are_missing(monkeypatch) -> None:
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)

    assert send_many_from_environment(["one", "two"]) == 0
