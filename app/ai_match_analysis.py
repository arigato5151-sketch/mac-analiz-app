"""Google Gemini commentary grounded in the app's statistical forecast."""

from __future__ import annotations

import json
import math
import time
from typing import Any

import pandas as pd
import requests


class MatchAnalysisAIError(RuntimeError):
    """A safe, user-facing error for AI commentary generation."""


def _recent_team_scores(
    history: pd.DataFrame, team_id: int, kickoff: pd.Timestamp, league_id: int,
    venue_filter: str | None = None,
) -> list[dict[str, Any]]:
    if history.empty:
        return []
    sample = history[
        (history["league_id"] == league_id)
        & (history["match_date"] < kickoff)
        & (
            (history["home_team_id"] == team_id)
            | (history["away_team_id"] == team_id)
        )
    ]
    if venue_filter == "home":
        sample = sample[sample["home_team_id"] == team_id]
    elif venue_filter == "away":
        sample = sample[sample["away_team_id"] == team_id]
    sample = sample.sort_values(["match_date", "id"], ascending=False).head(5)
    recent: list[dict[str, Any]] = []
    for match in sample.to_dict("records"):
        is_home = int(match["home_team_id"]) == team_id
        scored = int(match["home_score"] if is_home else match["away_score"])
        conceded = int(match["away_score"] if is_home else match["home_score"])
        result = "G" if scored > conceded else "B" if scored == conceded else "M"
        home_halftime = _optional_number(match.get("home_halftime_score"))
        away_halftime = _optional_number(match.get("away_halftime_score"))
        halftime = None
        if home_halftime is not None and away_halftime is not None:
            halftime = {
                "for": int(home_halftime if is_home else away_halftime),
                "against": int(away_halftime if is_home else home_halftime),
            }
        recent.append(
            {
                "date": pd.Timestamp(match["match_date"]).strftime("%Y-%m-%d"),
                "venue": "iç saha" if is_home else "deplasman",
                "score_for_against": f"{scored}-{conceded}",
                "result": result,
                "halftime_goals": halftime,
                "xg_for": _optional_number(
                    match.get("home_xg" if is_home else "away_xg")
                ),
                "xg_against": _optional_number(
                    match.get("away_xg" if is_home else "home_xg")
                ),
                "xa_for": _optional_number(
                    match.get("home_xa" if is_home else "away_xa")
                ),
                "xa_against": _optional_number(
                    match.get("away_xa" if is_home else "home_xa")
                ),
                "corners_for": _optional_number(
                    match.get("home_corners" if is_home else "away_corners")
                ),
                "corners_against": _optional_number(
                    match.get("away_corners" if is_home else "home_corners")
                ),
            }
        )
    return recent


def _optional_number(value: Any) -> float | None:
    """Convert a stored statistic to a finite number without imputing gaps."""
    if value is None or pd.isna(value):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _summarize_recent_team_matches(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize only observed values and report each metric's sample count."""
    metric_keys = {
        "goals_for_per_match": "scored",
        "goals_against_per_match": "conceded",
        "first_half_goals_for_per_match": "halftime_for",
        "first_half_goals_against_per_match": "halftime_against",
        "xg_for_per_match": "xg_for",
        "xg_against_per_match": "xg_against",
        "goals_minus_xg_per_match": "goals_minus_xg",
        "xa_for_per_match": "xa_for",
        "xa_against_per_match": "xa_against",
        "corners_for_per_match": "corners_for",
        "corners_against_per_match": "corners_against",
    }
    values: dict[str, list[float]] = {key: [] for key in metric_keys.values()}
    wins = draws = losses = points = 0
    for match in matches:
        scored, conceded = (int(part) for part in match["score_for_against"].split("-"))
        values["scored"].append(float(scored))
        values["conceded"].append(float(conceded))
        if match["result"] == "G":
            wins += 1
            points += 3
        elif match["result"] == "B":
            draws += 1
            points += 1
        else:
            losses += 1
        halftime = match.get("halftime_goals")
        if halftime:
            values["halftime_for"].append(float(halftime["for"]))
            values["halftime_against"].append(float(halftime["against"]))
        for key in (
            "xg_for", "xg_against", "xa_for", "xa_against",
            "corners_for", "corners_against",
        ):
            if match.get(key) is not None:
                values[key].append(float(match[key]))
        if match.get("xg_for") is not None:
            values["goals_minus_xg"].append(
                float(scored) - float(match["xg_for"])
            )

    summary: dict[str, Any] = {
        "matches": len(matches),
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "points_per_match": points / len(matches) if matches else None,
    }
    for output_key, source_key in metric_keys.items():
        observations = values[source_key]
        summary[output_key] = (
            sum(observations) / len(observations) if observations else None
        )
        summary[f"{source_key}_sample_size"] = len(observations)
    return summary


def _model_markets(forecast: dict[str, Any], home: str, away: str) -> dict[str, float]:
    matrix = forecast.get("score_matrix") or []

    def chance(predicate: Any) -> float:
        return sum(
            float(probability)
            for home_goals, line in enumerate(matrix)
            for away_goals, probability in enumerate(line)
            if predicate(home_goals, away_goals)
        )

    markets = {
        "1X": chance(lambda h, a: h >= a),
        "X2": chance(lambda h, a: a >= h),
        "12": chance(lambda h, a: h != a),
        f"Maç sonucu 1 · {home}": float(forecast["home_win"]),
        "Beraberlik": float(forecast["draw"]),
        f"Maç sonucu 2 · {away}": float(forecast["away_win"]),
        "0,5 gol üst": chance(lambda h, a: h + a >= 1),
        "0,5 gol alt": chance(lambda h, a: h + a == 0),
        "1,5 gol üst": chance(lambda h, a: h + a >= 2),
        "1,5 gol alt": chance(lambda h, a: h + a <= 1),
        "2,5 gol üst": chance(lambda h, a: h + a >= 3),
        "2,5 gol alt": chance(lambda h, a: h + a <= 2),
        "3,5 gol üst": chance(lambda h, a: h + a >= 4),
        "3,5 gol alt": chance(lambda h, a: h + a <= 3),
        "4,5 gol üst": chance(lambda h, a: h + a >= 5),
        "4,5 gol alt": chance(lambda h, a: h + a <= 4),
        "Karşılıklı gol · Var": float(forecast["btts"]),
        "Karşılıklı gol · Yok": 1.0 - float(forecast["btts"]),
        f"{home} 0,5 gol üst": chance(lambda h, a: h >= 1),
        f"{home} 0,5 gol alt": chance(lambda h, a: h == 0),
        f"{home} 1,5 gol üst": chance(lambda h, a: h >= 2),
        f"{home} 1,5 gol alt": chance(lambda h, a: h <= 1),
        f"{home} 2,5 gol üst": chance(lambda h, a: h >= 3),
        f"{away} 0,5 gol üst": chance(lambda h, a: a >= 1),
        f"{away} 0,5 gol alt": chance(lambda h, a: a == 0),
        f"{away} 1,5 gol üst": chance(lambda h, a: a >= 2),
        f"{away} 1,5 gol alt": chance(lambda h, a: a <= 1),
        f"{away} 2,5 gol üst": chance(lambda h, a: a >= 3),
        f"{home} kazanır ve KG Yok": chance(lambda h, a: h > a and a == 0),
        f"{away} kazanır ve KG Yok": chance(lambda h, a: a > h and h == 0),
        f"{home} kazanır ve 2,5 üst": chance(lambda h, a: h > a and h + a >= 3),
        f"{away} kazanır ve 2,5 üst": chance(lambda h, a: a > h and h + a >= 3),
    }
    scorelines = sorted(
        (
            (float(probability), home_goals, away_goals)
            for home_goals, score_row in enumerate(matrix)
            for away_goals, probability in enumerate(score_row)
        ),
        reverse=True,
    )[:5]
    for probability, home_goals, away_goals in scorelines:
        markets[f"Tam skor {home_goals}-{away_goals}"] = probability
    return markets


def _historical_model_evaluation(forecast: dict[str, Any]) -> dict[str, Any]:
    """Expose held-out model quality separately from this fixture's forecast."""
    market_metrics = forecast.get("tuning_market_metrics") or {}
    return {
        "held_out_matches": int(forecast.get("tuning_matches") or 0),
        "result_log_loss": _optional_number(forecast.get("tuning_log_loss")),
        "baseline_result_log_loss": _optional_number(
            forecast.get("tuning_baseline_log_loss")
        ),
        "markets": {
            key: {
                "sample_size": int(value.get("sample_size") or 0),
                "log_loss": _optional_number(value.get("model_log_loss")),
                "baseline_log_loss": _optional_number(value.get("baseline_log_loss")),
                "brier_score": _optional_number(value.get("model_brier_score")),
                "baseline_brier_score": _optional_number(
                    value.get("baseline_brier_score")
                ),
                "calibration_error": _optional_number(
                    value.get("model_calibration_error")
                ),
            }
            for key, value in market_metrics.items()
            if isinstance(value, dict)
        },
    }


def _score_distribution_summary(forecast: dict[str, Any]) -> dict[str, Any]:
    matrix = forecast.get("score_matrix") or []
    scores = sorted(
        (
            (float(probability), home_goals, away_goals)
            for home_goals, score_row in enumerate(matrix)
            for away_goals, probability in enumerate(score_row)
        ),
        reverse=True,
    )
    top_three = scores[:3]
    result_probs = {
        "home_win": float(forecast["home_win"]),
        "draw": float(forecast["draw"]),
        "away_win": float(forecast["away_win"]),
    }
    ranked_results = sorted(result_probs.items(), key=lambda item: item[1], reverse=True)
    return {
        "top_exact_scores": [
            {"score": f"{home_goals}-{away_goals}", "probability": probability}
            for probability, home_goals, away_goals in scores[:5]
        ],
        "top_three_exact_score_mass": sum(item[0] for item in top_three),
        "most_likely_result": ranked_results[0][0],
        "most_likely_result_probability": ranked_results[0][1],
        "result_probability_gap": ranked_results[0][1] - ranked_results[1][1],
    }


def _compare_team_summaries(
    home: dict[str, Any], away: dict[str, Any]
) -> dict[str, Any]:
    comparisons = (
        ("points_per_match", "recent_points_per_match_gap"),
        ("goals_for_per_match", "recent_goals_for_per_match_gap"),
        ("goals_against_per_match", "recent_goals_against_per_match_gap"),
        ("xg_for_per_match", "recent_xg_for_per_match_gap"),
        ("goals_minus_xg_per_match", "recent_goals_minus_xg_gap"),
    )
    result: dict[str, Any] = {}
    for metric, label in comparisons:
        home_value = _optional_number(home.get(metric))
        away_value = _optional_number(away.get(metric))
        result[label] = {
            "home": home_value,
            "away": away_value,
            "home_minus_away": (
                home_value - away_value
                if home_value is not None and away_value is not None
                else None
            ),
            "home_sample_size": int(home.get("matches") or 0)
            if metric in {
                "points_per_match",
                "goals_for_per_match",
                "goals_against_per_match",
            }
            else int(home.get(f"{metric.removesuffix('_per_match')}_sample_size") or 0),
            "away_sample_size": int(away.get("matches") or 0)
            if metric in {
                "points_per_match",
                "goals_for_per_match",
                "goals_against_per_match",
            }
            else int(away.get(f"{metric.removesuffix('_per_match')}_sample_size") or 0),
        }
    return result


def _recent_team_trend(matches: list[dict[str, Any]]) -> dict[str, Any]:
    """Compare the latest three results with the preceding two, when available."""
    if len(matches) < 5:
        return {"available": False, "matches": len(matches)}
    recent = _summarize_recent_team_matches(matches[:3])
    earlier = _summarize_recent_team_matches(matches[3:5])
    metrics = (
        "points_per_match",
        "goals_for_per_match",
        "goals_against_per_match",
        "xg_for_per_match",
        "xg_against_per_match",
    )
    sample_keys = {
        "points_per_match": "matches",
        "goals_for_per_match": "scored_sample_size",
        "goals_against_per_match": "conceded_sample_size",
        "xg_for_per_match": "xg_for_sample_size",
        "xg_against_per_match": "xg_against_sample_size",
    }
    changes = {}
    for metric in metrics:
        recent_value = _optional_number(recent.get(metric))
        earlier_value = _optional_number(earlier.get(metric))
        changes[metric] = {
            "recent_minus_earlier": (
                recent_value - earlier_value
                if recent_value is not None and earlier_value is not None
                else None
            ),
            "recent_sample_size": int(recent.get(sample_keys[metric]) or 0),
            "earlier_sample_size": int(earlier.get(sample_keys[metric]) or 0),
        }
    return {
        "available": True,
        "recent_window": recent,
        "earlier_window": earlier,
        "changes": changes,
    }


def _confirmed_lineup_context(row: dict[str, Any]) -> dict[str, Any] | None:
    records = row.get("fixture_lineups") or []
    if isinstance(records, str):
        try:
            records = json.loads(records)
        except json.JSONDecodeError:
            return None
    if not isinstance(records, list):
        return None
    kickoff = pd.to_datetime(row.get("kickoff") or row.get("match_date"), utc=True, errors="coerce")
    if pd.isna(kickoff):
        return None
    team_names = {
        int(row["home_team_id"]): str(row.get("home_name") or "Ev sahibi"),
        int(row["away_team_id"]): str(row.get("away_name") or "Deplasman"),
    }
    available = []
    for lineup in records:
        if not isinstance(lineup, dict):
            continue
        try:
            team_id = int(lineup["team_id"])
            confirmed_at = pd.to_datetime(lineup.get("confirmed_at"), utc=True, errors="coerce")
        except (KeyError, TypeError, ValueError):
            continue
        if team_id not in team_names or pd.isna(confirmed_at) or confirmed_at >= kickoff:
            continue
        starters = lineup.get("starters") or []
        substitutes = lineup.get("substitutes") or []
        if isinstance(starters, str):
            try:
                starters = json.loads(starters)
            except json.JSONDecodeError:
                starters = []
        if isinstance(substitutes, str):
            try:
                substitutes = json.loads(substitutes)
            except json.JSONDecodeError:
                substitutes = []
        available.append(
            {
                "team": team_names[team_id],
                "formation": lineup.get("formation"),
                "coach": lineup.get("coach_name"),
                "starters": starters[:20] if isinstance(starters, list) else [],
                "substitutes": substitutes[:20] if isinstance(substitutes, list) else [],
                "confirmed_at": confirmed_at.isoformat(),
            }
        )
    return {"source": "fixture_lineups", "teams": available} if available else None


def build_match_analysis_context(
    row: dict[str, Any], history: pd.DataFrame, forecast: dict[str, Any]
) -> dict[str, Any]:
    home = str(row.get("home_name") or "Ev sahibi")
    away = str(row.get("away_name") or "Deplasman")
    kickoff = pd.to_datetime(row.get("kickoff"), utc=True, errors="coerce")
    league_id = int(row["league_id"])
    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
    home_recent = (
        _recent_team_scores(history, home_id, kickoff, league_id)
        if not pd.isna(kickoff) else []
    )
    away_recent = (
        _recent_team_scores(history, away_id, kickoff, league_id)
        if not pd.isna(kickoff) else []
    )
    home_venue_recent = (
        _recent_team_scores(history, home_id, kickoff, league_id, venue_filter="home")
        if not pd.isna(kickoff) else []
    )
    away_venue_recent = (
        _recent_team_scores(history, away_id, kickoff, league_id, venue_filter="away")
        if not pd.isna(kickoff) else []
    )
    home_summary = _summarize_recent_team_matches(home_recent)
    away_summary = _summarize_recent_team_matches(away_recent)
    home_venue_summary = _summarize_recent_team_matches(home_venue_recent)
    away_venue_summary = _summarize_recent_team_matches(away_venue_recent)
    return {
        "match": {
            "home": home,
            "away": away,
            "competition": str(row.get("league_label") or row.get("league_name") or ""),
            "kickoff": kickoff.isoformat() if not pd.isna(kickoff) else None,
            "venue": str(row.get("venue") or "Bilinmiyor"),
        },
        "model": {
            "method": "Independent Poisson model based on league and recent team scores",
            "most_likely_score": str(forecast["most_likely_score"]),
            "expected_goals": {
                home: float(forecast["expected_home"]),
                away: float(forecast["expected_away"]),
            },
            "probabilities": {
                "home_win": float(forecast["home_win"]),
                "draw": float(forecast["draw"]),
                "away_win": float(forecast["away_win"]),
                "over_2_5": float(forecast["over_2_5"]),
                "both_teams_to_score": float(forecast["btts"]),
            },
            "score_distribution": _score_distribution_summary(forecast),
            "candidate_markets": _model_markets(forecast, home, away),
            "league_history_matches": int(forecast.get("history_count", 0)),
            "historical_model_evaluation": _historical_model_evaluation(forecast),
            "home_recent_summary": home_summary,
            "away_recent_summary": away_summary,
            "home_recent_trend": _recent_team_trend(home_recent),
            "away_recent_trend": _recent_team_trend(away_recent),
            "home_venue_summary": home_venue_summary,
            "away_venue_summary": away_venue_summary,
            "recent_form_comparisons": _compare_team_summaries(
                home_summary, away_summary
            ),
            "venue_form_comparisons": _compare_team_summaries(
                home_venue_summary, away_venue_summary
            ),
            "expected_goals_comparison": {
                "home": float(forecast["expected_home"]),
                "away": float(forecast["expected_away"]),
                "home_minus_away": float(forecast["expected_home"])
                - float(forecast["expected_away"]),
                "total": float(forecast["expected_home"])
                + float(forecast["expected_away"]),
            },
            "home_recent_matches": home_recent,
            "away_recent_matches": away_recent,
        },
        "confirmed_lineups": _confirmed_lineup_context(row),
        "available_lineup_or_injury_information": row.get("verified_team_news"),
    }


def generate_match_analysis(
    context: dict[str, Any], *, api_key: str, model: str = "gemini-3.8-flash",
    instructions_override: str | None = None, json_mode: bool = False,
    max_output_tokens: int = 1800,
    response_schema: dict[str, Any] | None = None,
) -> str:
    """Generate a Turkish analysis using only supplied match and model data."""
    if not api_key:
        raise MatchAnalysisAIError("Google API anahtarı yapılandırılmamış.")
    instructions = (
        "Bir futbol analisti gibi Türkçe yaz. Verilen JSON, maç ve model hakkında "
        "kullanılabilecek tek bilgi kaynağıdır. Oyuncu, sakatlık, diziliş, teknik "
        "direktör, canlı haber, oran veya form bilgisi uydurma. Veri eksikse bunu "
        "confirmed_lineups alanında kayıt varsa yalnızca oradaki teknik direktör, "
        "diziliş ve oyuncu adlarını kesinleşmiş bilgi olarak kullan; alan boşsa kadro "
        "ve sakatlıklar hakkında çıkarım yapma. "
        "açıkça söyle. İstatistiksel olasılıkları kesinlik gibi sunma; verilen "
        "olasılıkları değiştirme. Taktik fikirleri veriyle desteklenen çıkarım veya "
        "koşullu senaryo olarak belirt. Takımın son maçlarında score_for_against "
        "takımın attığı-yediği goldür; G/B/M galibiyet/beraberlik/mağlubiyet demektir. "
        "home_recent_matches ve away_recent_matches geçmişte gözlenen maçları içerir; "
        "xG, xA ve korner değerleri yalnızca kaydı bulunan maçlar içindir, null değerleri "
        "tahmin etme. Özetlerdeki her *_sample_size ilgili ölçümün kaç maçtan geldiğini "
        "gösterir. Bu geçmiş xG değerlerini modelin beklenen gol tahminiyle karıştırma. "
        "Form yorumunu bu sayısal örneklere bağla; örneklem küçükse bunu belirt. "
        "Mümkün olduğunda form bölümünde son maçların G/B/M sayısını ve maç başı puanı; "
        "hücum-savunma bölümünde atılan/yenen gol ortalamasını ve mevcutsa xG farkını "
        "sayılarıyla kullan. Bir çıkarımın dayandığı ölçümün örnek sayısı azsa kesin "
        "dille yazma. xG, xA ve korner tek başına pres, topa sahip olma veya diziliş "
        "kanıtı değildir; bunlardan yalnızca sınırlı hücum üretimi çıkarımı yap. "
        "home_venue_summary ev sahibinin son iç saha, away_venue_summary deplasmanın "
        "son dış saha maçlarını özetler. Örnek sayısını mutlaka dikkate al; üçten azsa "
        "bu saha ayrımını güçlü form kanıtı sayma. goals_minus_xg pozitifse goller xG'yi "
        "aşmış, negatifse altında kalmıştır; küçük örneklemde bunu sürdürülebilir kalite "
        "diye sunma. "
        "score_distribution içindeki top_exact_scores tekil tam skor olasılıklarıdır; "
        "top_three_exact_score_mass ilk üç skorun toplamıdır. En olası tam skorun tek başına "
        "yüksek olasılıklı olduğunu varsayma. Geniş 1X2 ve gol marketleriyle tek skor "
        "olasılığını ayır; maç senaryolarını bu dağılım ve olasılık farkına göre kur. "
        "Dakika veya erken gol gibi zamanlama iddiası üretme; model zamanlama tahmini yapmıyor. "
        "recent_form_comparisons ve venue_form_comparisons içindeki home_minus_away farkı "
        "ev sahibi eksi deplasmandır; null farkı hesaplama. En belirgin iki farklılaşan "
        "sayısal sinyali seçip takım isimleri ve örnek sayılarıyla açıkla. Her maçta aynı "
        "genel giriş ve klişe senaryoları tekrarlama; veri desteklemiyorsa favori veya oyun "
        "kontrolü iddiası kurma. "
        "home_recent_trend ve away_recent_trend varsa son 3 maç ile önceki 2 maçın "
        "değişimini gösterir. Yalnızca available true ise trendden söz et; örneklemler "
        "küçük olduğundan bunu kısa dönem sinyali diye nitele, kesin eğilim sayma. "
        "historical_model_evaluation bu maçın tahmini değil, ayrılmış geçmiş maçlarda "
        "yapılmış model kontrolüdür. Yalnızca held_out_matches en az 30 ise kısaca "
        "değin ve sonucu ön değerlendirme olarak nitele; örnek sayısını belirt, düşük "
        "log-loss/Brier değerinin daha iyi olduğunu unutma ve bu ölçümleri maç olasılığı "
        "gibi sunma. Örnek azsa performans hükmü verme. "
        "Markdown kullan ve şu başlıklarla özgün, maç özelinde bir yorum üret: "
        "Maçın profili; İki takımın form sinyalleri; Olası oyun planları ve eşleşmeler; "
        "Maçın kırılma noktaları; Farklı oyun senaryoları; Model olasılıklarına göre "
        "temkinli ve dengeli seçimler. Her seçimde verilen model olasılığını "
        "yüzdeyle yaz; listede olmayan market/olasılık ekleme. Oran, kazanç garantisi "
        "veya kesin sonuç iddiası verme. Sonunda tahminin sınırlarını kısa belirt."
    )
    if instructions_override:
        instructions = instructions_override
    request_url = (
        f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
    )
    request_headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }
    request_body = {
        "system_instruction": {"parts": [{"text": instructions}]},
        "contents": [{
            "role": "user",
            "parts": [{
                "text": json.dumps(context, ensure_ascii=False, separators=(",", ":"))
            }],
        }],
        "generationConfig": {
            "maxOutputTokens": max_output_tokens,
            "temperature": 0.7,
            **({"responseMimeType": "application/json"} if json_mode else {}),
            **({"responseSchema": response_schema} if response_schema else {}),
        },
    }
    try:
        for attempt in range(3):
            response = requests.post(
                request_url,
                headers=request_headers,
                json=request_body,
                timeout=60,
            )
            if response.status_code != 503 or attempt == 2:
                break
            retry_after = response.headers.get("Retry-After", "")
            try:
                delay = min(max(float(retry_after), 1.0), 8.0)
            except ValueError:
                delay = 1.5 * (attempt + 1)
            time.sleep(delay)
    except requests.RequestException as exc:
        if model != "gemini-3.5-flash-lite":
            return generate_match_analysis(
                context, api_key=api_key, model="gemini-3.5-flash-lite",
                instructions_override=instructions_override, json_mode=json_mode,
                max_output_tokens=max_output_tokens, response_schema=response_schema,
            )
        raise MatchAnalysisAIError("Google Gemini API bağlantısı kurulamadı.") from exc
    if not response.ok:
        # Google returns a structured reason (for example, model not found or
        # API not enabled). Include that safe diagnostic so configuration
        # errors can be fixed without exposing request headers or the API key.
        try:
            error_payload = response.json().get("error", {})
            detail = str(error_payload.get("message", "")).strip()
            status = str(error_payload.get("status", "")).strip()
        except (ValueError, AttributeError):
            detail, status = "", ""
        if (
            response.status_code == 404
            and model != "gemini-3.8-flash"
            and "no longer available" in detail.lower()
        ):
            return generate_match_analysis(
                context, api_key=api_key, model="gemini-3.8-flash",
                instructions_override=instructions_override, json_mode=json_mode,
                max_output_tokens=max_output_tokens, response_schema=response_schema,
            )
        if response.status_code in {429, 503} and model != "gemini-3.5-flash-lite":
            return generate_match_analysis(
                context, api_key=api_key, model="gemini-3.5-flash-lite",
                instructions_override=instructions_override, json_mode=json_mode,
                max_output_tokens=max_output_tokens, response_schema=response_schema,
            )
        diagnostic = ": ".join(part for part in (status, detail) if part)
        if len(diagnostic) > 400:
            diagnostic = diagnostic[:397] + "..."
        suffix = f" — {diagnostic}" if diagnostic else ""
        raise MatchAnalysisAIError(
            f"Google Gemini API isteği başarısız oldu (HTTP {response.status_code}){suffix}"
        )
    try:
        payload = response.json()
        candidates = payload.get("candidates") or []
        parts = candidates[0].get("content", {}).get("parts", []) if candidates else []
        output = "\n".join(
            str(part.get("text", "")).strip() for part in parts if part.get("text")
        ).strip()
    except (ValueError, AttributeError) as exc:
        raise MatchAnalysisAIError("Google Gemini API yanıtı okunamadı.") from exc
    if not output:
        raise MatchAnalysisAIError("Google Gemini API boş bir analiz döndürdü.")
    return output


def generate_match_risk_recommendations(
    context: dict[str, Any], *, api_key: str, model: str = "gemini-3.8-flash"
) -> dict[str, list[dict[str, Any]]]:
    """Ask Gemini to choose varied selections for two moderate risk profiles."""
    allowed = context.get("model", {}).get("candidate_markets", {})
    instructions = (
        "Türkçe futbol tahmin analisti olarak yalnızca verilen maç ve model JSON'una dayan. "
        "İki risk profili için seçim yap: low temkinli ve yüksek olasılıklı; medium dengeli "
        "ve orta olasılıklı. Yüksek riskli veya çok düşük olasılıklı seçim üretme. Yalnızca "
        "candidate_markets içindeki tam market adlarını kullan; olasılık uydurma veya değiştirme. "
        "Her profile farklı market türlerinden iki seçim ver, aynı türü profilde tekrar etme. "
        "İki profil arasında aynı marketi tekrarlama. Low seçimleri en az %60, medium seçimleri "
        "%30-%72 olasılık bandında olsun. Her seçim için maç verisine dayalı kısa Türkçe gerekçe ver. "
        "İstenen JSON: {\"low\":[{\"market\":\"market adı\",\"reason\":\"kısa gerekçe\"}],"
        "\"medium\":[{\"market\":\"market adı\",\"reason\":\"kısa gerekçe\"}]} ."
    )
    raw = generate_match_analysis(
        context, api_key=api_key, model=model,
        instructions_override=instructions, json_mode=True,
    )
    try:
        result = json.loads(raw)
    except ValueError as exc:
        raise MatchAnalysisAIError("Yapay zekâ risk seçimlerini yapılandırılmış biçimde döndürmedi.") from exc
    validated = _validate_risk_selections(result, allowed)
    if not any(validated.values()):
        raise MatchAnalysisAIError("Yapay zekâ geçerli risk seçimi döndürmedi.")
    return validated


def generate_match_ai_package(
    context: dict[str, Any], *, api_key: str, model: str = "gemini-3.8-flash"
) -> dict[str, Any]:
    """Generate the instant-display commentary and risk picks in one AI call."""
    instructions = (
        "Türkçe, maç özelinde ve yalnızca verilen JSON'a dayanarak analiz yaz. Oyuncu, sakatlık, "
        "diziliş veya haber uydurma; confirmed_lineups varsa yalnızca oradaki bilgileri kullan, "
        "boşsa kadro/sakatlık yorumu yapma. Model olasılıklarını değiştirme, kesinlik "
        "iddiası ve bahis oranı verme. Geçmiş maçlardaki xG, xA, korner ve devre skorları gözlenen "
        "verilerdir; null değerleri doldurma, özetlerdeki sample_size alanlarıyla veri kapsamını "
        "kontrol et ve geçmiş xG'yi model beklenen golünden ayrı tut. "
        "Form ve üretim yorumlarında mümkünse son maç G/B/M toplamını, maç başı puanı, "
        "gol ortalamasını ve mevcut xG ortalamasını somut rakamlarla belirt. Örneklem "
        "küçükse bunu söyle ve kesin konuşma. xG, xA veya kornerden diziliş, pres ya da "
        "topa sahip olma iddiası çıkarma. Her taktik senaryonun bir veri dayanağını ve "
        "hangi koşulda gerçekleşeceğini açıkla; dayanak yoksa bunu varsayım olarak etiketle. "
        "home_venue_summary ev sahibinin iç saha, away_venue_summary deplasman formudur. "
        "matches sayısını kontrol et; üçten az maç varsa bu saha ayrımı sınırlıdır. "
        "goals_minus_xg gözlenen gol ile xG farkıdır, sürdürülebilir beceri kanıtı değildir; "
        "az örnekle kesin yorumlama. "
        "score_distribution içindeki tam skor yüzdeleri tek tek skorların olasılığıdır; "
        "ilk sıradaki skor kesin veya yüksek olasılıklı demek değildir. Üç skorun toplam "
        "payını ve geniş maç sonucu/gol olasılıklarını ayır; senaryoları bunlara göre kur. "
        "Model dakika bazlı olmadığı için erken gol veya belirli dakika senaryosu uydurma. "
        "recent_form_comparisons ve venue_form_comparisons farkları ev sahibi eksi "
        "deplasmandır. Null veya yetersiz örnekli veriyi yorumlama. Her maçta en ayırt edici "
        "iki sinyali seçip isimler ve örnek sayılarıyla anlat; genel futbol klişelerini ve "
        "verinin desteklemediği favori/oyun kontrolü iddialarını kullanma. "
        "home_recent_trend ve away_recent_trend içindeki değişim son 3 maçı önceki 2 "
        "maçla karşılaştırır. available false ise trend çıkarma; true olsa bile küçük "
        "örneklem nedeniyle yalnızca kısa dönem işareti olarak aktar. "
        "historical_model_evaluation yalnızca ayrılmış geçmiş maçlardaki model performansıdır; "
        "held_out_matches 30'un altındaysa başarı kanıtı olarak kullanma. En az 30 maç varsa "
        "bunu ön değerlendirme olarak tanımla, sayıyı ve düşük log-loss/Brier'ın anlamını "
        "kısaca belirt; bu ölçümü bu maçın kazanma olasılığıyla karıştırma. "
        "Maçın profili, form sinyalleri, oyun planları/eşleşmeler, "
        "kırılma noktaları ve farklı maç senaryolarını içeren kısa ama anlamlı bir Markdown yorum "
        "oluştur. Ayrıca candidate_markets içinden tam adları seçerek iki risk grubuna ikişer tahmin "
        "öner: low temkinli ve yüksek olasılıklı; medium dengeli ve orta olasılıklı. Yüksek riskli "
        "seçim üretme. Seçimleri farklı market türlerinden yap, her biri için kısa gerekçe yaz. "
        "Aynı marketi gruplar arasında tekrar etme. "
        "Market adı veya olasılığı uydurma. Yalnızca şu JSON'u üret: {\"analysis\":\"Markdown metni\","
        "\"risk_selections\":{\"low\":[{\"market\":\"tam market adı\",\"reason\":\"gerekçe\"}],"
        "\"medium\":[...]}}."
    )
    raw = generate_match_analysis(
        context,
        api_key=api_key,
        model=model,
        instructions_override=instructions,
        json_mode=True,
    )
    try:
        package = json.loads(raw)
    except ValueError as exc:
        raise MatchAnalysisAIError("Yapay zekâ analiz paketini okuyamadı.") from exc
    analysis = str(package.get("analysis", "")).strip() if isinstance(package, dict) else ""
    if not analysis:
        raise MatchAnalysisAIError("Yapay zekâ analiz metni döndürmedi.")
    package["analysis"] = analysis[:12000]
    package["risk_selections"] = _validate_risk_selections(
        package.get("risk_selections", {}),
        context.get("model", {}).get("candidate_markets", {}),
    )
    if not any(package["risk_selections"].values()):
        raise MatchAnalysisAIError("Yapay zekâ geçerli risk seçimi döndürmedi.")
    return package


def generate_match_ai_packages(
    contexts: list[dict[str, Any]], *, api_key: str, model: str = "gemini-3.8-flash"
) -> dict[int, dict[str, Any]]:
    """Generate compact, match-specific packages in one quota-efficient request."""
    if not contexts:
        return {}
    instructions = (
        "Türkçe futbol analisti olarak her maçı ayrı değerlendir ve yalnızca verilen "
        "JSON'daki verilere dayan. Oyuncu, sakatlık, diziliş, oran veya haber uydurma. "
        "confirmed_lineups alanında kayıt varsa yalnızca oradaki teknik direktör, diziliş "
        "ve oyuncu adlarını kullan; boşsa kadro/sakatlık hakkında yorum yapma. "
        "Geçmiş xG/xA/korner ve devre skorları gözlemdir; null veriyi uydurma, ilgili "
        "sample_size alanlarını dikkate al ve geçmiş xG'yi model beklenen golünden ayır. "
        "Her takımın formunu mümkün olduğunda son maç G/B/M toplamı ve maç başı puanıyla; "
        "hücum üretimini gol ve mevcut xG ortalamalarıyla açıkla. Bu istatistiklerin "
        "örnek sayısı azsa güven düzeyini düşür. Yalnızca xG/xA/kornerden diziliş, pres "
        "veya topa sahip olma iddiası üretme. Taktik senaryoları veriye bağlı koşullu "
        "çıkarım olarak yaz; veri yoksa varsayım olduğunu açıkça belirt. "
        "home_venue_summary ev sahibinin, away_venue_summary deplasmanın saha özelindeki "
        "formudur. Üçten az maç varsa örneklemin sınırlı olduğunu belirt. goals_minus_xg "
        "farkını yakın dönem gözlemi olarak yorumla; az örnekle kalıcı hücum becerisi çıkarma. "
        "score_distribution içindeki tam skor değerleri tekil skor olasılıklarıdır; "
        "en olası skoru kesin sonuç gibi anlatma. İlk üç skorun toplam payıyla 1X2 ve gol "
        "olasılıklarını birbirine karıştırma. Model zamanlama üretmediği için dakika/erken gol "
        "iddiasında bulunma. recent_form_comparisons ve venue_form_comparisons içindeki "
        "farklar ev sahibi eksi deplasmandır; null değerleri yorumlama. O maçta en belirgin "
        "iki sayısal farkı takım adları ve örnek sayılarıyla vurgula. Her maç için aynı genel "
        "girişleri tekrarlama ve istatistik desteklemediğinde baskın taraf iddiası kurma. "
        "home_recent_trend ve away_recent_trend son 3 maçı önceki 2 maçla karşılaştırır. "
        "Yalnızca available true ise trend kullan; sonucu küçük örneklemli kısa dönem "
        "sinyali olarak niteler, kesin eğilim diye sunma. "
        "historical_model_evaluation bu maça ait değildir, ayrılmış geçmiş değerlendirme "
        "örneklemidir. En az 30 maç yoksa model güvenilirliği hakkında hüküm verme. Yeterli "
        "örnek varsa sonucu ön değerlendirme olarak tanımla, sayıyı belirt ve kalite metriğini "
        "maç olasılığından ayrı tut. "
        "Her maç için 4 kısa başlıklı Markdown yorum yaz: Maç profili, Form ve eşleşme, "
        "Olası senaryolar, Risk profillerine göre seçimler. Yorum her maçta o maçın "
        "takım adlarına, formuna ve Poisson olasılıklarına özgü olsun; veri zayıfsa "
        "bunu belirt. Seçimleri yalnızca ilgili maçın candidate_markets listesinden "
        "seç ve market olasılıklarını değiştirme. Her risk grubuna farklı türlerden iki seçim "
        "gerekçe koy: low temkinli, medium dengeli. Yüksek riskli seçim üretme. Aynı maçta "
        "marketleri tekrar etme. Kesinlik veya kazanç garantisi verme. Her girişteki "
        "match_id değerini aynen koru. Yalnızca bu JSON şemasını döndür: "
        "{\"matches\":[{\"match_id\":123,\"analysis\":\"...\","
        "\"risk_selections\":{\"low\":[{\"market\":\"...\",\"reason\":\"...\"}],"
        "\"medium\":[...]}}]}"
    )
    selection_schema = {
        "type": "ARRAY",
        "items": {
            "type": "OBJECT",
            "properties": {
                "market": {"type": "STRING"},
                "reason": {"type": "STRING"},
            },
            "required": ["market", "reason"],
        },
    }
    response_schema = {
        "type": "OBJECT",
        "properties": {
            "matches": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "match_id": {"type": "INTEGER"},
                        "analysis": {"type": "STRING"},
                        "risk_selections": {
                            "type": "OBJECT",
                            "properties": {
                                "low": selection_schema,
                                "medium": selection_schema,
                            },
                            "required": ["low", "medium"],
                        },
                    },
                    "required": ["match_id", "analysis", "risk_selections"],
                },
            },
        },
        "required": ["matches"],
    }
    raw = generate_match_analysis(
        {"matches": contexts}, api_key=api_key, model=model,
        instructions_override=instructions, json_mode=True, max_output_tokens=12000,
        response_schema=response_schema,
    )
    try:
        payload = json.loads(raw)
    except ValueError as exc:
        raise MatchAnalysisAIError("Yapay zekâ maç paketlerini JSON olarak döndürmedi.") from exc
    context_by_id = {
        int(item["match_id"]): item
        for item in contexts
        if item.get("match_id") is not None
    }
    packages: dict[int, dict[str, Any]] = {}
    matches = payload.get("matches", []) if isinstance(payload, dict) else []
    for item in matches if isinstance(matches, list) else []:
        if not isinstance(item, dict):
            continue
        try:
            match_id = int(item["match_id"])
        except (KeyError, TypeError, ValueError):
            continue
        context = context_by_id.get(match_id)
        analysis = str(item.get("analysis", "")).strip()
        if context is None or not analysis:
            continue
        selections = _validate_risk_selections(
            item.get("risk_selections", {}),
            context.get("model", {}).get("candidate_markets", {}),
        )
        packages[match_id] = {
            "analysis": analysis[:12000],
            "risk_selections": selections,
        }
    if not packages:
        raise MatchAnalysisAIError("Yapay zekâ geçerli maç paketi döndürmedi.")
    return packages


def _validate_risk_selections(
    result: Any, allowed: Any
) -> dict[str, list[dict[str, Any]]]:
    validated: dict[str, list[dict[str, Any]]] = {"low": [], "medium": []}
    used: set[str] = set()
    if not isinstance(result, dict) or not isinstance(allowed, dict):
        return validated
    for profile in validated:
        items = result.get(profile, [])
        if not isinstance(items, list):
            continue
        used_families: set[str] = set()
        for item in items[:2]:
            if not isinstance(item, dict):
                continue
            market = str(item.get("market", "")).strip()
            reason = str(item.get("reason", "")).strip()
            try:
                probability = float(allowed.get(market, -1)) if market in allowed else -1
            except (TypeError, ValueError):
                continue
            if not math.isfinite(probability) or not 0 <= probability <= 1:
                continue
            family = _market_family(market)
            in_band = (
                probability >= 0.60
                if profile == "low"
                else 0.30 <= probability <= 0.72
            )
            if (
                market in allowed
                and market not in used
                and family not in used_families
                and reason
                and in_band
            ):
                validated[profile].append(
                    {
                        "market": market,
                        "reason": reason[:300],
                        "model_probability": probability,
                    }
                )
                used.add(market)
                used_families.add(family)
    return validated


def _market_family(market: str) -> str:
    label = market.casefold()
    if label.startswith("tam skor"):
        return "score"
    if "karşılıklı gol" in label:
        return "btts"
    if "kazanır ve" in label:
        return "combined"
    if label in {"1x", "x2", "12"}:
        return "double_chance"
    if "gol" in label and any(
        line in label for line in ("0,5", "1,5", "2,5", "3,5", "4,5")
    ):
        if label.startswith(("0,5 ", "1,5 ", "2,5 ", "3,5 ", "4,5 ")):
            return "total_goals"
        return "team_goals"
    return "match_result"
