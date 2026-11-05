"""Persist nightly forecast and recent-form data for fast match-card rendering."""

from __future__ import annotations

import hashlib
import json
import math
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd

from app.main import (
    ANALYSIS_MODEL_VERSION,
    build_precomputed_match_analysis,
    tune_recent_form_prior,
)
from app.ai_match_analysis import (
    MatchAnalysisAIError,
    build_match_analysis_context,
    confirmed_lineup_context,
    generate_match_ai_packages,
    iddaa_odds_revision,
    previous_match_ids_for_fixture,
    previous_lineup_context,
)
from config.settings import get_optional_setting, get_supabase_admin_settings
from db.db_client import DatabaseError, SupabaseRestClient

MATCH_COLUMNS = (
    "id,league_id,home_team_id,away_team_id,match_date,status"
)
HISTORY_COLUMNS = (
    "id,league_id,home_team_id,away_team_id,match_date,status,home_score,away_score,"
    "home_halftime_score,away_halftime_score,home_xg,away_xg,home_xa,away_xa,"
    "home_corners,away_corners,home_shots,away_shots,home_shots_on_target,away_shots_on_target,"
    "home_possession,away_possession"
)
NUMERIC_HISTORY_COLUMNS = (
    "id",
    "league_id",
    "home_team_id",
    "away_team_id",
    "home_score",
    "away_score",
    "home_halftime_score",
    "away_halftime_score",
    "home_xg",
    "away_xg",
    "home_xa",
    "away_xa",
    "home_corners",
    "away_corners",
    "home_shots",
    "away_shots",
    "home_shots_on_target",
    "away_shots_on_target",
    "home_possession",
    "away_possession",
)


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if hasattr(value, "item"):
        return _json_safe(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is pd.NA or value is pd.NaT:
        return None
    return value


def _load_league_history(db: SupabaseRestClient, league_id: int) -> pd.DataFrame:
    records = db.select_all(
        "matches",
        columns=HISTORY_COLUMNS,
        filters={
            "status": "eq.finished",
            "home_score": "not.is.null",
            "away_score": "not.is.null",
            "league_id": f"eq.{league_id}",
        },
        order="match_date.desc,id.desc",
    )
    history = pd.DataFrame(records)
    if history.empty:
        return history
    history["match_date"] = pd.to_datetime(
        history["match_date"], utc=True, errors="coerce"
    )
    for column in NUMERIC_HISTORY_COLUMNS:
        if column not in history:
            history[column] = pd.NA
        history[column] = pd.to_numeric(history[column], errors="coerce")
    return history.dropna(subset=["match_date"])


def _load_commentary_feedback_summary(
    db: SupabaseRestClient, *, now: datetime
) -> dict[str, Any] | None:
    """Aggregate recent anonymous feedback without loading identities or comment text."""
    since = (now - pd.Timedelta(days=30)).isoformat()
    try:
        feedback = db.select_all(
            "ai_analysis_feedback",
            columns="helpful,issue_codes",
            filters={"created_at": f"gte.{since}"},
            order="created_at.desc,id.desc",
        )
    except DatabaseError as exc:
        print(f"AI commentary feedback lookup skipped: {type(exc).__name__}")
        return None
    if len(feedback) < 5:
        return None
    helpful_count = sum(bool(row.get("helpful")) for row in feedback)
    issue_counts: dict[str, int] = {}
    allowed = {"generic", "missing_data", "unclear_reasoning", "data_error"}
    for row in feedback:
        codes = row.get("issue_codes") or []
        if not isinstance(codes, list):
            continue
        for code in set(codes) & allowed:
            issue_counts[code] = issue_counts.get(code, 0) + 1
    return {
        "period_days": 30,
        "sample_size": len(feedback),
        "helpful_rate": helpful_count / len(feedback),
        "issue_counts": issue_counts,
    }


def _commentary_lineup_revision(row: dict[str, Any]) -> str:
    """Hash lineup content while ignoring polling timestamps."""
    sources = (
        confirmed_lineup_context(row),
        previous_lineup_context(row),
        previous_lineup_context(row, include_confirmed=True),
    )
    normalized = []
    for source in sources:
        teams = []
        for team in (source or {}).get("teams", []):
            teams.append(
                {
                    key: team.get(key)
                    for key in (
                        "team", "formation", "coach", "starters", "substitutes",
                        "previous_match_id", "previous_match_date", "previous_competition",
                    )
                }
            )
        normalized.append(teams)
    payload = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def collect_match_analysis_cache(
    db: SupabaseRestClient, *, now: datetime | None = None
) -> list[dict[str, Any]]:
    """Build analysis payloads for all upcoming scheduled fixtures."""
    reference = now or datetime.now(timezone.utc)
    if reference.tzinfo is None:
        raise ValueError("now must be timezone-aware")
    reference = reference.astimezone(timezone.utc)
    cache_until = reference + pd.Timedelta(hours=36)
    fixtures = db.select_all(
        "matches",
        columns=MATCH_COLUMNS,
        filters={
            "status": "eq.scheduled",
            "match_date": f"gt.{reference.isoformat()}",
            "and": f"(match_date.lt.{cache_until.isoformat()})",
        },
        order="match_date.asc,id.asc",
    )
    commentary_feedback = _load_commentary_feedback_summary(db, now=reference)
    feedback_revision = hashlib.sha256(
        json.dumps(commentary_feedback, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()
    if commentary_feedback:
        print(
            "Recent AI commentary feedback summary: "
            f"{commentary_feedback['sample_size']} votes; aggregate style guidance enabled."
        )
    cached_ai: dict[int, dict[str, Any]] = {}
    fixture_ids = [str(item["id"]) for item in fixtures if item.get("id") is not None]
    latest_odds: dict[int, dict[str, Any]] = {}
    for offset in range(0, len(fixture_ids), 100):
        chunk = fixture_ids[offset : offset + 100]
        if not chunk:
            continue
        quote_rows = db.select_all(
            "odds_quote_history",
            columns="id,match_id,bookmaker,odds,captured_at,source_updated_at",
            filters={
                "bookmaker": "eq.iddaa.com",
                "match_id": f"in.({','.join(chunk)})",
            },
            order="captured_at.asc,id.asc",
        )
        for quote in quote_rows:
            try:
                latest_odds[int(quote["match_id"])] = quote
            except (KeyError, TypeError, ValueError):
                continue
    lineups_by_match: dict[int, list[dict[str, Any]]] = {}
    for offset in range(0, len(fixture_ids), 100):
        chunk = fixture_ids[offset : offset + 100]
        if not chunk:
            continue
        try:
            lineup_rows = db.select_all(
                "fixture_lineups",
                columns=(
                    "match_id,team_id,formation,coach_name,starters,substitutes,confirmed_at"
                ),
                filters={"match_id": f"in.({','.join(chunk)})"},
                order="match_id.asc,team_id.asc",
            )
        except DatabaseError as exc:
            print(f"Confirmed lineup lookup skipped: {type(exc).__name__}")
            lineups_by_match = {}
            break
        for lineup in lineup_rows:
            try:
                lineups_by_match.setdefault(int(lineup["match_id"]), []).append(lineup)
            except (KeyError, TypeError, ValueError):
                continue
    for offset in range(0, len(fixture_ids), 100):
        cached_rows = db.select_all(
            "match_analysis_cache",
            columns="match_id,model_version,analysis_json",
            filters={"match_id": f"in.({','.join(fixture_ids[offset:offset + 100])})"},
            order="match_id.asc",
        )
        for cached in cached_rows:
            cached_analysis = cached.get("analysis_json") or {}
            if (
                cached.get("model_version") == ANALYSIS_MODEL_VERSION
                and isinstance(cached_analysis, dict)
                and cached_analysis.get("ai_analysis")
                and cached_analysis.get("commentary_feedback_revision") == feedback_revision
            ):
                cached_ai[int(cached["match_id"])] = {
                    "analysis": cached_analysis["ai_analysis"],
                    "risk_selections": cached_analysis.get("ai_risk_selections") or {},
                    "lineup_revision": cached_analysis.get("commentary_lineup_revision"),
                    "odds_revision": cached_analysis.get("commentary_odds_revision"),
                }
    team_names = {
        int(item["id"]): str(item["name"])
        for item in db.select_all("teams", columns="id,name")
        if item.get("id") is not None and item.get("name")
    }
    league_names = {
        int(item["id"]): str(item["name"])
        for item in db.select_all("leagues", columns="id,name")
        if item.get("id") is not None and item.get("name")
    }
    google_api_key = get_optional_setting("GOOGLE_API_KEY")
    google_model = get_optional_setting("GOOGLE_MODEL", "gemini-3.8-flash")
    ai_failures = 0
    if not google_api_key:
        print("GOOGLE_API_KEY is missing; nightly AI commentary will not be generated.")
    histories: dict[int, pd.DataFrame] = {}
    tuning_by_league: dict[int, dict[str, Any] | None] = {}
    rows: list[dict[str, Any]] = []
    cache_row_by_match: dict[int, dict[str, Any]] = {}
    ai_contexts: list[dict[str, Any]] = []
    analysis_by_match: dict[int, dict[str, Any]] = {}
    for fixture in fixtures:
        try:
            league_id = int(fixture["league_id"])
            match_id = int(fixture["id"])
            home_team_id = int(fixture["home_team_id"])
            away_team_id = int(fixture["away_team_id"])
            match_date = pd.to_datetime(fixture["match_date"], utc=True, errors="coerce")
            if pd.isna(match_date):
                continue
        except (KeyError, TypeError, ValueError):
            continue

        if league_id not in histories:
            histories[league_id] = _load_league_history(db, league_id)
            tuning_by_league[league_id] = (
                tune_recent_form_prior(histories[league_id])
                if not histories[league_id].empty
                else None
            )
        row = {
            **fixture,
            "id": match_id,
            "league_id": league_id,
            "home_team_id": home_team_id,
            "away_team_id": away_team_id,
            "match_date": match_date,
            "kickoff": match_date,
            "home_name": team_names.get(home_team_id, f"Ev sahibi {home_team_id}"),
            "away_name": team_names.get(away_team_id, f"Deplasman {away_team_id}"),
            "league_name": league_names.get(league_id, ""),
            "league_label": league_names.get(league_id, ""),
            "fixture_lineups": lineups_by_match.get(match_id, []),
            "iddaa_odds_quote": latest_odds.get(match_id),
        }
        try:
            prior_matches = db.select(
                "matches",
                columns=(
                    "id,league_id,home_team_id,away_team_id,match_date,status,"
                    "home_score,away_score"
                ),
                filters={
                    "status": "eq.finished",
                    "home_score": "not.is.null",
                    "away_score": "not.is.null",
                    "match_date": f"lt.{match_date.isoformat()}",
                    "or": (
                        f"(home_team_id.in.({home_team_id},{away_team_id}),"
                        f"away_team_id.in.({home_team_id},{away_team_id}))"
                    ),
                },
                limit=100,
                order="match_date.desc,id.desc",
            )
            prior_match_ids = previous_match_ids_for_fixture(
                row, pd.DataFrame(prior_matches)
            )
            prior_lineups = []
            if prior_match_ids:
                prior_lineups = db.select_all(
                    "fixture_lineups",
                    columns=(
                        "match_id,team_id,formation,coach_name,starters,substitutes,confirmed_at"
                    ),
                    filters={
                        "match_id": f"in.({','.join(map(str, sorted(set(prior_match_ids.values()))))})"
                    },
                    order="match_id.asc,team_id.asc",
                )
            prior_dates = {
                int(item["id"]): pd.to_datetime(item["match_date"], utc=True).isoformat()
                for item in prior_matches
                if item.get("id") is not None and item.get("match_date") is not None
            }
            prior_competitions = {
                int(item["id"]): league_names.get(int(item["league_id"]))
                for item in prior_matches
                if item.get("id") is not None and item.get("league_id") is not None
            }
            row["previous_fixture_lineups"] = [
                lineup | {
                    "previous_match_date": prior_dates.get(int(lineup["match_id"])),
                    "previous_competition": prior_competitions.get(int(lineup["match_id"])),
                }
                for lineup in prior_lineups
            ]
        except DatabaseError as exc:
            print(f"Previous lineup lookup skipped for match_id={match_id}: {type(exc).__name__}")
            row["previous_fixture_lineups"] = []
        analysis = build_precomputed_match_analysis(
            row, histories[league_id], tuning_by_league[league_id]
        )
        if google_api_key and analysis.get("forecast"):
            analysis_by_match[match_id] = analysis
            lineup_revision = _commentary_lineup_revision(row)
            odds_revision = iddaa_odds_revision(latest_odds.get(match_id))
            if (
                match_id in cached_ai
                and cached_ai[match_id].get("lineup_revision") == lineup_revision
                and cached_ai[match_id].get("odds_revision") == odds_revision
            ):
                ai_package = cached_ai[match_id]
                analysis["ai_analysis"] = ai_package["analysis"]
                analysis["ai_risk_selections"] = ai_package["risk_selections"]
            else:
                ai_context = build_match_analysis_context(
                    row,
                    histories[league_id],
                    analysis["forecast"],
                    half_forecasts=analysis.get("halves"),
                    commentary_quality_feedback=commentary_feedback,
                    odds_quote=latest_odds.get(match_id),
                )
                ai_context["match_id"] = match_id
                ai_context["model"]["candidate_markets"] = {
                    str(name): float(probability)
                    for name, probability in ai_context["model"]["candidate_markets"].items()
                }
                ai_contexts.append(ai_context)
            analysis["commentary_feedback_revision"] = feedback_revision
            analysis["commentary_lineup_revision"] = lineup_revision
            analysis["commentary_odds_revision"] = iddaa_odds_revision(
                latest_odds.get(match_id)
            )
        cache_row = {
                "match_id": match_id,
                "league_id": league_id,
                "home_team_id": home_team_id,
                "away_team_id": away_team_id,
                "match_date": match_date.isoformat(),
                "model_version": ANALYSIS_MODEL_VERSION,
                "calculated_at": reference.isoformat(),
                "analysis_json": _json_safe(analysis),
            }
        rows.append(cache_row)
        cache_row_by_match[match_id] = cache_row
    # One structured request can analyze several fixtures, keeping a full slate
    # within the free-tier daily request budget. Existing AI results are reused
    # by the second scheduled refresh instead of consuming the quota twice.
    for offset in range(0, len(ai_contexts), 10):
        batch = ai_contexts[offset : offset + 10]
        try:
            packages = generate_match_ai_packages(
                batch, api_key=google_api_key or "", model=google_model
            )
            for match_id, package in packages.items():
                analysis = analysis_by_match.get(match_id)
                if analysis is not None:
                    analysis["ai_analysis"] = package["analysis"]
                    analysis["ai_risk_selections"] = package["risk_selections"]
                    cache_row = cache_row_by_match.get(match_id)
                    if cache_row is not None:
                        cache_row["analysis_json"]["ai_analysis"] = package["analysis"]
                        cache_row["analysis_json"]["ai_risk_selections"] = package[
                            "risk_selections"
                        ]
            missing = {int(context["match_id"]) for context in batch} - packages.keys()
            ai_failures += len(missing)
            if missing:
                print(f"AI precompute returned no package for matches: {sorted(missing)}")
        except MatchAnalysisAIError as exc:
            ai_failures += len(batch)
            print(
                f"AI precompute failed for match batch "
                f"{offset // 10 + 1} ({len(batch)} matches): {exc}"
            )
    return rows


def main() -> None:
    settings = get_supabase_admin_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    rows = collect_match_analysis_cache(db)
    db.delete(
        "match_analysis_cache",
        filters={"match_date": f"lt.{datetime.now(timezone.utc).isoformat()}"},
    )
    for offset in range(0, len(rows), 100):
        db.upsert(
            "match_analysis_cache",
            rows[offset : offset + 100],
            on_conflict="match_id",
        )
    print(
        json.dumps(
            {
                "fixtures_cached": len(rows),
                "ai_analyses_cached": sum(
                    1 for row in rows if row["analysis_json"].get("ai_analysis")
                ),
                "ai_failures": sum(
                    1
                    for row in rows
                    if row["analysis_json"].get("forecast")
                    and not row["analysis_json"].get("ai_analysis")
                ),
                "model_version": ANALYSIS_MODEL_VERSION,
            }
        )
    )


if __name__ == "__main__":
    main()
