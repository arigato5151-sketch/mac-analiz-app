from __future__ import annotations

from notifications.final_results import final_result_message


def test_final_result_message_marks_each_market_correct_or_incorrect() -> None:
    message = final_result_message(
        {"home_score": 2, "away_score": 1},
        {
            "prob_home_win": 0.60,
            "prob_draw": 0.20,
            "prob_away_win": 0.20,
            "prob_over_2_5": 0.55,
            "prob_btts": 0.45,
        },
        home_team="Ev",
        away_team="Deplasman",
        league_name="Lig",
    )

    assert "Ev 2 — 1 Deplasman" in message
    assert "Tahmin sonuçları" in message
    assert "1-X-2: Ev kazanır %60 ✓" in message
    assert "Üst 2.5: Üst (%55) ✓" in message
    assert "KG Var: KG Yok (%55) ✗" in message
