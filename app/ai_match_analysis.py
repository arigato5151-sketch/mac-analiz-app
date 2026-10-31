"""Google Gemini commentary grounded in the app's statistical forecast."""

from __future__ import annotations

import json
import time
from typing import Any

import pandas as pd
import requests


class MatchAnalysisAIError(RuntimeError):
    """A safe, user-facing error for AI commentary generation."""


def _recent_team_scores(
    history: pd.DataFrame, team_id: int, kickoff: pd.Timestamp, league_id: int
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
    ].sort_values(["match_date", "id"], ascending=False).head(5)
    recent: list[dict[str, Any]] = []
    for match in sample.to_dict("records"):
        is_home = int(match["home_team_id"]) == team_id
        scored = int(match["home_score"] if is_home else match["away_score"])
        conceded = int(match["away_score"] if is_home else match["home_score"])
        result = "G" if scored > conceded else "B" if scored == conceded else "M"
        recent.append(
            {
                "date": pd.Timestamp(match["match_date"]).strftime("%Y-%m-%d"),
                "venue": "iç saha" if is_home else "deplasman",
                "score_for_against": f"{scored}-{conceded}",
                "result": result,
            }
        )
    return recent


def _model_markets(forecast: dict[str, Any], home: str, away: str) -> dict[str, float]:
    matrix = forecast.get("score_matrix") or []

    def chance(predicate: Any) -> float:
        return sum(
            float(probability)
            for home_goals, line in enumerate(matrix)
            for away_goals, probability in enumerate(line)
            if predicate(home_goals, away_goals)
        )

    return {
        "1X": chance(lambda h, a: h >= a),
        "X2": chance(lambda h, a: a >= h),
        "12": chance(lambda h, a: h != a),
        f"Maç sonucu 1 · {home}": float(forecast["home_win"]),
        "Beraberlik": float(forecast["draw"]),
        f"Maç sonucu 2 · {away}": float(forecast["away_win"]),
        "1,5 gol üst": chance(lambda h, a: h + a >= 2),
        "2,5 gol üst": chance(lambda h, a: h + a >= 3),
        "2,5 gol alt": chance(lambda h, a: h + a <= 2),
        "3,5 gol alt": chance(lambda h, a: h + a <= 3),
        "Karşılıklı gol · Var": float(forecast["btts"]),
        "Karşılıklı gol · Yok": 1.0 - float(forecast["btts"]),
        f"{home} 0,5 gol üst": chance(lambda h, a: h >= 1),
        f"{away} 0,5 gol üst": chance(lambda h, a: a >= 1),
        f"{home} kazanır ve KG Yok": chance(lambda h, a: h > a and a == 0),
        f"{away} kazanır ve KG Yok": chance(lambda h, a: a > h and h == 0),
        f"{home} kazanır ve 2,5 üst": chance(lambda h, a: h > a and h + a >= 3),
        f"{away} kazanır ve 2,5 üst": chance(lambda h, a: a > h and h + a >= 3),
        f"Tam skor {forecast['most_likely_score']}": chance(
            lambda h, a: f"{h}-{a}" == str(forecast["most_likely_score"]).replace("–", "-").replace(":", "-")
        ),
    }


def build_match_analysis_context(
    row: dict[str, Any], history: pd.DataFrame, forecast: dict[str, Any]
) -> dict[str, Any]:
    home = str(row.get("home_name") or "Ev sahibi")
    away = str(row.get("away_name") or "Deplasman")
    kickoff = pd.to_datetime(row.get("kickoff"), utc=True, errors="coerce")
    league_id = int(row["league_id"])
    home_id = int(row["home_team_id"])
    away_id = int(row["away_team_id"])
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
            "candidate_markets": _model_markets(forecast, home, away),
            "league_history_matches": int(forecast.get("history_count", 0)),
            "home_recent_matches": _recent_team_scores(
                history, home_id, kickoff, league_id
            ) if not pd.isna(kickoff) else [],
            "away_recent_matches": _recent_team_scores(
                history, away_id, kickoff, league_id
            ) if not pd.isna(kickoff) else [],
        },
        "available_lineup_or_injury_information": row.get("verified_team_news"),
    }


def generate_match_analysis(
    context: dict[str, Any], *, api_key: str, model: str = "gemini-3.8-flash",
    instructions_override: str | None = None, json_mode: bool = False,
) -> str:
    """Generate a Turkish analysis using only supplied match and model data."""
    if not api_key:
        raise MatchAnalysisAIError("Google API anahtarı yapılandırılmamış.")
    instructions = (
        "Bir futbol analisti gibi Türkçe yaz. Verilen JSON, maç ve model hakkında "
        "kullanılabilecek tek bilgi kaynağıdır. Oyuncu, sakatlık, diziliş, teknik "
        "direktör, canlı haber, oran veya form bilgisi uydurma. Veri eksikse bunu "
        "açıkça söyle. İstatistiksel olasılıkları kesinlik gibi sunma; verilen "
        "olasılıkları değiştirme. Taktik fikirleri veriyle desteklenen çıkarım veya "
        "koşullu senaryo olarak belirt. Takımın son maçlarında score_for_against "
        "takımın attığı-yediği goldür; G/B/M galibiyet/beraberlik/mağlubiyet demektir. "
        "Markdown kullan ve şu başlıklarla özgün, maç özelinde bir yorum üret: "
        "Maçın profili; İki takımın form sinyalleri; Olası oyun planları ve eşleşmeler; "
        "Maçın kırılma noktaları; Farklı oyun senaryoları; Model olasılıklarına göre "
        "düşük/orta/yüksek riskli seçimler. Her seçimde verilen model olasılığını "
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
            "maxOutputTokens": 1800,
            "temperature": 0.7,
            **({"responseMimeType": "application/json"} if json_mode else {}),
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
            )
        if response.status_code == 503 and model != "gemini-3.6-flash":
            return generate_match_analysis(
                context, api_key=api_key, model="gemini-3.6-flash",
                instructions_override=instructions_override, json_mode=json_mode,
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
) -> dict[str, list[dict[str, str]]]:
    """Ask Gemini to choose match-specific selections for three risk profiles."""
    allowed = context.get("model", {}).get("candidate_markets", {})
    instructions = (
        "Türkçe futbol tahmin analisti olarak yalnızca verilen maç ve model JSON'una dayan. "
        "Üç risk profili için market seç: low daha yüksek olasılıklı ve temkinli; medium "
        "makul olasılık ile getiri dengesinde; high düşük olasılıklı fakat futbol açısından "
        "tutarlı ve daha koşullu. Sadece candidate_markets içindeki tam market adlarını "
        "seç; olasılık üretme veya değiştirme. Her profile 1-2 seçim koy, aynı seçimi iki "
        "profile tekrar etme. Her seçim için maç verisine dayanan tek cümlelik Türkçe "
        "gerekçe ver. Çıktı yalnızca şu JSON şemasında olsun: {\"low\":[{\"market\":\"...\","
        "\"reason\":\"...\"}],\"medium\":[...],\"high\":[...]}."
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
        "diziliş veya haber uydurma; veride yoksa belirt. Model olasılıklarını değiştirme, kesinlik "
        "iddiası ve bahis oranı verme. Maçın profili, form sinyalleri, oyun planları/eşleşmeler, "
        "kırılma noktaları ve farklı maç senaryolarını içeren kısa ama anlamlı bir Markdown yorum "
        "oluştur. Ayrıca candidate_markets içinden tam adları seçerek üç risk grubuna 1-2 tahmin "
        "öner: low daha olası/temkinli; medium dengeli; high daha koşullu/yüksek risk. Her seçim "
        "için veriye dayalı tek cümle Türkçe gerekçe yaz. Aynı marketi gruplar arasında tekrar etme. "
        "Market adı veya olasılığı uydurma. Yalnızca şu JSON'u üret: {\"analysis\":\"Markdown metni\","
        "\"risk_selections\":{\"low\":[{\"market\":\"tam market adı\",\"reason\":\"gerekçe\"}],"
        "\"medium\":[...],\"high\":[...]}}."
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


def _validate_risk_selections(
    result: Any, allowed: Any
) -> dict[str, list[dict[str, str]]]:
    validated: dict[str, list[dict[str, str]]] = {"low": [], "medium": [], "high": []}
    used: set[str] = set()
    if not isinstance(result, dict) or not isinstance(allowed, dict):
        return validated
    for profile in validated:
        items = result.get(profile, [])
        if not isinstance(items, list):
            continue
        for item in items[:2]:
            if not isinstance(item, dict):
                continue
            market = str(item.get("market", "")).strip()
            reason = str(item.get("reason", "")).strip()
            if market in allowed and market not in used and reason:
                validated[profile].append({"market": market, "reason": reason[:300]})
                used.add(market)
    return validated
