import pandas as pd

from app.components.data import _has_suspicious_result_quote, _latest_quote_per_match


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


def test_malformed_underround_result_quote_is_rejected():
    assert _has_suspicious_result_quote(
        {"home_win": "17.50", "draw": "17.50", "away_win": "3.73"}
    )


def test_normal_result_quote_is_kept():
    assert not _has_suspicious_result_quote(
        {"home_win": "1.03", "draw": "17.50", "away_win": "51.00"}
    )
