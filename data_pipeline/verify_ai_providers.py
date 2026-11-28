"""Validate AI providers with a real cached fixture without writing to Supabase."""

from __future__ import annotations

import time
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from app.ai_match_analysis import (
    MatchAnalysisAIError,
    _compact_fallback_match_context,
    build_compatible_match_analysis_context,
    generate_match_ai_packages,
)
from app.main import ANALYSIS_MODEL_VERSION
from config.settings import get_optional_setting, get_supabase_admin_settings
from data_pipeline.precompute_match_analysis import _load_commentary_feedback_summary
from db.db_client import DatabaseError, SupabaseRestClient


def _api_key(name: str) -> str:
    return "".join(get_optional_setting(name).replace("\ufeff", "").split())


def _sample_match_context(db: SupabaseRestClient) -> tuple[int, dict[str, Any]]:
    """Build the production commentary context for one upcoming cached fixture."""
    now = datetime.now(timezone.utc)
    fixtures = db.select(
        "matches",
        columns="id,league_id,home_team_id,away_team_id,match_date,status",
        filters={"status": "eq.scheduled", "match_date": f"gt.{now.isoformat()}"},
        limit=100,
        order="match_date.asc,id.asc",
    )
    if not fixtures:
        raise MatchAnalysisAIError("Canlı doğrulama için yaklaşan maç bulunamadı.")

    fixture_ids = [str(row["id"]) for row in fixtures if row.get("id") is not None]
    caches = db.select_all(
        "match_analysis_cache",
        columns="match_id,league_id,home_team_id,away_team_id,match_date,model_version,analysis_json",
        filters={
            "match_id": f"in.({','.join(fixture_ids)})",
            "model_version": f"eq.{ANALYSIS_MODEL_VERSION}",
        },
        order="match_date.asc,match_id.asc",
    )
    cache_by_id = {int(row["match_id"]): row for row in caches}

    fixture = next(
        (
            row
            for row in fixtures
            if int(row["id"]) in cache_by_id
            and isinstance(cache_by_id[int(row["id"])].get("analysis_json"), dict)
            and isinstance(
                cache_by_id[int(row["id"])]["analysis_json"].get("forecast"), dict
            )
        ),
        None,
    )
    if fixture is None:
        raise MatchAnalysisAIError(
            "Yaklaşan maçlar için güncel model tahmini önbelleği bulunamadı."
        )

    match_id = int(fixture["id"])
    cache = cache_by_id[match_id]
    analysis = cache["analysis_json"]
    home_id = int(fixture["home_team_id"])
    away_id = int(fixture["away_team_id"])
    league_id = int(fixture["league_id"])
    team_rows = db.select_all(
        "teams",
        columns="id,name",
        filters={"id": f"in.({home_id},{away_id})"},
    )
    league_rows = db.select_all(
        "leagues", columns="id,name", filters={"id": f"eq.{league_id}"}
    )
    team_names = {int(row["id"]): str(row["name"]) for row in team_rows}
    league_name = str(league_rows[0]["name"]) if league_rows else ""

    row = {
        **fixture,
        "id": match_id,
        "league_id": league_id,
        "home_team_id": home_id,
        "away_team_id": away_id,
        "match_date": pd.to_datetime(fixture["match_date"], utc=True),
        "kickoff": pd.to_datetime(fixture["match_date"], utc=True),
        "home_name": team_names.get(home_id, f"Ev sahibi {home_id}"),
        "away_name": team_names.get(away_id, f"Deplasman {away_id}"),
        "league_name": league_name,
        "league_label": league_name,
        "fixture_lineups": [],
        "previous_fixture_lineups": [],
    }
    try:
        row["fixture_lineups"] = db.select_all(
            "fixture_lineups",
            columns="match_id,team_id,formation,coach_name,starters,substitutes,confirmed_at",
            filters={"match_id": f"eq.{match_id}"},
            order="match_id.asc,team_id.asc",
        )
    except DatabaseError:
        # A missing optional lineup table must not prevent the core AI check.
        pass

    quotes = db.select_all(
        "odds_quote_history",
        columns="id,match_id,bookmaker,odds,captured_at,source_updated_at",
        filters={"bookmaker": "eq.iddaa.com", "match_id": f"eq.{match_id}"},
        order="captured_at.asc,id.asc",
    )
    history = pd.DataFrame(analysis.get("history") or [])
    if not history.empty:
        history["match_date"] = pd.to_datetime(
            history["match_date"], utc=True, errors="coerce"
        )
        for column in (
            "id", "league_id", "home_team_id", "away_team_id", "home_score",
            "away_score", "home_halftime_score", "away_halftime_score", "home_xg",
            "away_xg", "home_xa", "away_xa", "home_corners", "away_corners",
            "home_shots", "away_shots", "home_shots_on_target",
            "away_shots_on_target", "home_possession", "away_possession",
        ):
            if column in history:
                history[column] = pd.to_numeric(history[column], errors="coerce")

    context = build_compatible_match_analysis_context(
        row,
        history,
        analysis["forecast"],
        half_forecasts=analysis.get("halves"),
        commentary_quality_feedback=_load_commentary_feedback_summary(db, now=now),
        odds_quote=quotes[-1] if quotes else None,
    )
    context["match_id"] = match_id
    context["model"]["candidate_markets"] = {
        str(name): float(probability)
        for name, probability in context["model"]["candidate_markets"].items()
    }
    return match_id, _compact_fallback_match_context(context)


def _provider_configs() -> tuple[dict[str, Any], ...]:
    return (
        {
            "name": "Groq",
            "provider": "openai_compatible",
            "api_key": _api_key("GROQ_API_KEY"),
            "model": get_optional_setting("GROQ_MODEL", "openai/gpt-oss-20b"),
            "base_url": "https://api.groq.com/openai/v1",
            "strict_json_schema": True,
            "reasoning_effort": "low",
            "max_output_tokens": 2400,
            "request_timeout_seconds": 20,
        },
        {
            "name": "NVIDIA NIM",
            "provider": "openai_compatible",
            "api_key": _api_key("NVIDIA_API_KEY"),
            "model": get_optional_setting(
                "NVIDIA_MODEL", "nvidia/nemotron-3-ultra-550b-a55b"
            ),
            "base_url": "https://integrate.api.nvidia.com/v1",
            "strict_json_schema": False,
            "max_output_tokens": 2600,
            "extra_body": {
                "chat_template_kwargs": {
                    "enable_thinking": False,
                    "force_nonempty_content": True,
                }
            },
            "request_timeout_seconds": 90,
        },
    )


def main() -> None:
    settings = get_supabase_admin_settings()
    db = SupabaseRestClient(settings.supabase_url, settings.supabase_service_role_key)
    match_id, context = _sample_match_context(db)
    print(f"Provider integration sample prepared (match_id={match_id}; no database writes).")

    failures = 0
    for provider in _provider_configs():
        name = provider["name"]
        if not provider["api_key"]:
            print(f"{name}: API key missing")
            failures += 1
            continue
        started = time.perf_counter()
        try:
            packages = generate_match_ai_packages(
                [context],
                api_key="",
                model=provider["model"],
                fallback_providers=[provider],
                fallback_delay_seconds=0,
            )
            package = packages.get(match_id)
            if not package or not package.get("analysis"):
                raise MatchAnalysisAIError("Gerçek maç bağlamı için yorum paketi dönmedi.")
            elapsed = time.perf_counter() - started
            print(
                f"{name}: full-prompt generation succeeded "
                f"({elapsed:.1f}s; {len(package['analysis'])} chars; no database write)."
            )
        except (MatchAnalysisAIError, ValueError, AttributeError) as exc:
            elapsed = time.perf_counter() - started
            print(f"{name}: generation failed after {elapsed:.1f}s: {exc}")
            failures += 1
    if failures:
        raise SystemExit(f"{failures} AI provider integration check(s) failed.")


if __name__ == "__main__":
    main()
