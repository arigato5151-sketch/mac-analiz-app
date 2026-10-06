import pandas as pd

from app.components.data import _latest_quote_per_match


def test_latest_quote_selection_uses_freshest_provider_not_nesine_priority():
    quotes = pd.DataFrame(
        [
            {
                "match_id": 1,
                "bookmaker": "Nesine",
                "odds": {"home_win": "17.50", "draw": "5.00", "away_win": "1.10"},
                "captured_at": pd.Timestamp("2026-10-06T13:32:00Z"),
            },
            {
                "match_id": 1,
                "bookmaker": "Bet365",
                "odds": {"home_win": "1.75", "draw": "3.50", "away_win": "4.80"},
                "captured_at": pd.Timestamp("2026-10-06T14:57:00Z"),
            },
        ]
    )

    selected = _latest_quote_per_match(quotes)

    assert len(selected) == 1
    assert selected.iloc[0]["bookmaker"] == "Bet365"
    assert selected.iloc[0]["odds"]["home_win"] == "1.75"
