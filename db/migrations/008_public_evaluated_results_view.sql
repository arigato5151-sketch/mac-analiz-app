-- The public audit view must not inherit private snapshot-table permissions.
BEGIN;

CREATE OR REPLACE VIEW public.evaluated_prediction_results
WITH (security_invoker = false)
AS
SELECT
    performance.prediction_id,
    performance.match_id,
    performance.actual_result,
    performance.was_correct,
    performance.brier_score,
    performance.evaluated_at,
    fixture.league_id,
    fixture.match_date,
    fixture.home_score,
    fixture.away_score,
    COALESCE(snapshot.prob_home_win, prediction.prob_home_win) AS prob_home_win,
    COALESCE(snapshot.prob_draw, prediction.prob_draw) AS prob_draw,
    COALESCE(snapshot.prob_away_win, prediction.prob_away_win) AS prob_away_win,
    COALESCE(snapshot.prob_over_2_5, prediction.prob_over_2_5) AS prob_over_2_5,
    COALESCE(snapshot.prob_btts, prediction.prob_btts) AS prob_btts,
    COALESCE(snapshot.model_version, prediction.model_version) AS model_version,
    COALESCE(snapshot.captured_at, prediction.predicted_at) AS predicted_at,
    home.name AS home_team,
    away.name AS away_team,
    league.name AS league_name,
    performance.over_2_5_actual,
    performance.over_2_5_was_correct,
    performance.over_2_5_brier_score,
    performance.btts_actual,
    performance.btts_was_correct,
    performance.btts_brier_score
FROM public.live_prediction_performance AS performance
JOIN public.matches AS fixture ON fixture.id = performance.match_id
JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
LEFT JOIN public.prediction_snapshots AS snapshot ON snapshot.id = performance.snapshot_id
LEFT JOIN public.teams AS home ON home.id = fixture.home_team_id
LEFT JOIN public.teams AS away ON away.id = fixture.away_team_id
LEFT JOIN public.leagues AS league ON league.id = fixture.league_id;

GRANT SELECT ON public.evaluated_prediction_results TO anon, service_role;

COMMIT;
