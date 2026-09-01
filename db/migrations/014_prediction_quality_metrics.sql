-- Persist proper-scoring-rule quality metrics and expose daily model aggregates.
BEGIN;

ALTER TABLE public.prediction_performance
    ADD COLUMN IF NOT EXISTS log_loss NUMERIC CHECK (log_loss IS NULL OR log_loss >= 0);

UPDATE public.prediction_performance AS performance
SET log_loss = -LN(GREATEST(
    0.000000000000001::NUMERIC,
    CASE performance.actual_result
        WHEN 'home_win' THEN COALESCE(
            (SELECT snapshot.prob_home_win FROM public.prediction_snapshots AS snapshot WHERE snapshot.id = performance.snapshot_id),
            prediction.prob_home_win
        )
        WHEN 'draw' THEN COALESCE(
            (SELECT snapshot.prob_draw FROM public.prediction_snapshots AS snapshot WHERE snapshot.id = performance.snapshot_id),
            prediction.prob_draw
        )
        WHEN 'away_win' THEN COALESCE(
            (SELECT snapshot.prob_away_win FROM public.prediction_snapshots AS snapshot WHERE snapshot.id = performance.snapshot_id),
            prediction.prob_away_win
        )
    END
))
FROM public.predictions AS prediction
WHERE prediction.id = performance.prediction_id
  AND performance.log_loss IS NULL;

CREATE OR REPLACE VIEW public.prediction_quality_daily
WITH (security_invoker = true)
AS
SELECT
    DATE(performance.evaluated_at) AS evaluated_day,
    COALESCE(snapshot.model_version, prediction.model_version) AS model_version,
    COUNT(*)::INTEGER AS sample_size,
    AVG(CASE WHEN performance.was_correct THEN 1.0 ELSE 0.0 END) AS accuracy,
    AVG(performance.brier_score) AS avg_brier_score,
    AVG(performance.log_loss) AS avg_log_loss
FROM public.prediction_performance AS performance
JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
LEFT JOIN public.prediction_snapshots AS snapshot ON snapshot.id = performance.snapshot_id
WHERE performance.log_loss IS NOT NULL
GROUP BY DATE(performance.evaluated_at), COALESCE(snapshot.model_version, prediction.model_version);

GRANT SELECT ON public.prediction_quality_daily TO anon, service_role;

COMMIT;
