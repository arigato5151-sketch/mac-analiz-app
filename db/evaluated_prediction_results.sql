-- Read model evaluations in one query for the public Streamlit results page.
-- security_invoker keeps the underlying table RLS policies in force for anon.
BEGIN;

CREATE OR REPLACE VIEW public.evaluated_prediction_results
WITH (security_invoker = true)
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
    prediction.prob_home_win,
    prediction.prob_draw,
    prediction.prob_away_win,
    prediction.prob_over_2_5,
    prediction.prob_btts,
    prediction.model_version,
    prediction.predicted_at,
    home.name AS home_team,
    away.name AS away_team,
    league.name AS league_name
FROM public.live_prediction_performance AS performance
JOIN public.matches AS fixture ON fixture.id = performance.match_id
JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
LEFT JOIN public.teams AS home ON home.id = fixture.home_team_id
LEFT JOIN public.teams AS away ON away.id = fixture.away_team_id
LEFT JOIN public.leagues AS league ON league.id = fixture.league_id;

GRANT SELECT ON public.evaluated_prediction_results TO anon, service_role;

COMMIT;
