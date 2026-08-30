-- One official prediction per match for public live-performance reporting.
-- Older model-version rows remain preserved for offline analysis but never
-- inflate the user-facing production scorecard.
BEGIN;

CREATE OR REPLACE VIEW public.live_prediction_performance
WITH (security_invoker = true)
AS
WITH ranked AS (
    SELECT
        performance.prediction_id,
        performance.match_id,
        performance.actual_result,
        performance.was_correct,
        performance.brier_score,
        performance.evaluated_at,
        performance.snapshot_id,
        performance.over_2_5_actual,
        performance.over_2_5_was_correct,
        performance.over_2_5_brier_score,
        performance.btts_actual,
        performance.btts_was_correct,
        performance.btts_brier_score,
        ROW_NUMBER() OVER (
            PARTITION BY performance.match_id
            ORDER BY performance.evaluated_at DESC, prediction.predicted_at DESC, prediction.id DESC
        ) AS row_rank
    FROM public.prediction_performance AS performance
    JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
)
SELECT
    prediction_id,
    match_id,
    actual_result,
    was_correct,
    brier_score,
    evaluated_at,
    snapshot_id,
    over_2_5_actual,
    over_2_5_was_correct,
    over_2_5_brier_score,
    btts_actual,
    btts_was_correct,
    btts_brier_score
FROM ranked
WHERE row_rank = 1;

GRANT SELECT ON public.live_prediction_performance TO anon, service_role;

COMMIT;
