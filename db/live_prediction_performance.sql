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
        ROW_NUMBER() OVER (
            PARTITION BY performance.match_id
            ORDER BY prediction.predicted_at DESC, prediction.id DESC
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
    evaluated_at
FROM ranked
WHERE row_rank = 1;

GRANT SELECT ON public.live_prediction_performance TO anon, service_role;

COMMIT;
