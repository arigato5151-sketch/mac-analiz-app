-- Publish aggregate shadow-model evidence without exposing raw predictions.
BEGIN;

CREATE OR REPLACE VIEW public.shadow_model_status
WITH (security_invoker = false)
AS
WITH latest_production AS (
    SELECT match_id, was_correct, brier_score
    FROM (
        SELECT
            performance.match_id,
            performance.was_correct,
            performance.brier_score,
            ROW_NUMBER() OVER (
                PARTITION BY performance.match_id
                ORDER BY performance.evaluated_at DESC, performance.prediction_id DESC
            ) AS row_rank
        FROM public.prediction_performance AS performance
    ) AS ranked
    WHERE row_rank = 1
), candidate_matches AS (
    SELECT
        candidate.model_version,
        candidate.status,
        candidate.registered_at,
        candidate.promoted_at,
        NULLIF(candidate.offline_metrics->>'log_loss', '')::DOUBLE PRECISION AS offline_log_loss,
        NULLIF(candidate.offline_metrics->>'raw_log_loss', '')::DOUBLE PRECISION AS offline_raw_log_loss,
        NULLIF(candidate.offline_metrics->>'baseline_log_loss', '')::DOUBLE PRECISION AS offline_baseline_log_loss,
        NULLIF(candidate.offline_metrics->>'expected_calibration_error', '')::DOUBLE PRECISION AS offline_ece,
        shadow.id AS shadow_prediction_id,
        shadow_performance.was_correct AS candidate_correct,
        shadow_performance.brier_score AS candidate_brier,
        production.was_correct AS production_correct,
        production.brier_score AS production_brier
    FROM public.model_candidates AS candidate
    LEFT JOIN public.shadow_predictions AS shadow
        ON shadow.model_version = candidate.model_version
    LEFT JOIN public.shadow_prediction_performance AS shadow_performance
        ON shadow_performance.shadow_prediction_id = shadow.id
    LEFT JOIN latest_production AS production
        ON production.match_id = shadow_performance.match_id
)
SELECT
    model_version,
    status,
    registered_at,
    promoted_at,
    offline_log_loss,
    offline_raw_log_loss,
    offline_baseline_log_loss,
    offline_ece,
    COUNT(shadow_prediction_id)::INTEGER AS prediction_count,
    COUNT(candidate_brier)::INTEGER AS evaluated_matches,
    COUNT(production_brier) FILTER (WHERE candidate_brier IS NOT NULL)::INTEGER AS paired_matches,
    AVG(candidate_correct::INTEGER) FILTER (WHERE production_brier IS NOT NULL) AS candidate_accuracy,
    AVG(candidate_brier) FILTER (WHERE production_brier IS NOT NULL) AS candidate_brier,
    AVG(production_correct::INTEGER) FILTER (WHERE candidate_brier IS NOT NULL) AS production_accuracy,
    AVG(production_brier) FILTER (WHERE candidate_brier IS NOT NULL) AS production_brier
FROM candidate_matches
GROUP BY
    model_version,
    status,
    registered_at,
    promoted_at,
    offline_log_loss,
    offline_raw_log_loss,
    offline_baseline_log_loss,
    offline_ece;

REVOKE ALL ON public.shadow_model_status FROM PUBLIC;
GRANT SELECT ON public.shadow_model_status TO anon, service_role;

COMMIT;
