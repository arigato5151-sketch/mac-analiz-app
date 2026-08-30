-- Persist auditable metrics for total-goals and both-teams-to-score markets.
BEGIN;

ALTER TABLE public.prediction_performance
    ADD COLUMN IF NOT EXISTS over_2_5_actual BOOLEAN,
    ADD COLUMN IF NOT EXISTS over_2_5_was_correct BOOLEAN,
    ADD COLUMN IF NOT EXISTS over_2_5_brier_score NUMERIC CHECK (over_2_5_brier_score IS NULL OR over_2_5_brier_score BETWEEN 0 AND 1),
    ADD COLUMN IF NOT EXISTS btts_actual BOOLEAN,
    ADD COLUMN IF NOT EXISTS btts_was_correct BOOLEAN,
    ADD COLUMN IF NOT EXISTS btts_brier_score NUMERIC CHECK (btts_brier_score IS NULL OR btts_brier_score BETWEEN 0 AND 1);

WITH source AS (
    SELECT
        performance.id,
        fixture.home_score + fixture.away_score >= 3 AS over_2_5_actual,
        fixture.home_score > 0 AND fixture.away_score > 0 AS btts_actual,
        COALESCE(snapshot.prob_over_2_5, prediction.prob_over_2_5) AS prob_over_2_5,
        COALESCE(snapshot.prob_btts, prediction.prob_btts) AS prob_btts
    FROM public.prediction_performance AS performance
    JOIN public.matches AS fixture ON fixture.id = performance.match_id
    JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
    LEFT JOIN public.prediction_snapshots AS snapshot ON snapshot.id = performance.snapshot_id
)
UPDATE public.prediction_performance AS performance
SET
    over_2_5_actual = source.over_2_5_actual,
    over_2_5_was_correct = CASE WHEN source.prob_over_2_5 IS NULL THEN NULL ELSE (source.prob_over_2_5 >= 0.5) = source.over_2_5_actual END,
    over_2_5_brier_score = CASE WHEN source.prob_over_2_5 IS NULL THEN NULL ELSE POWER(source.prob_over_2_5 - source.over_2_5_actual::INT, 2) END,
    btts_actual = source.btts_actual,
    btts_was_correct = CASE WHEN source.prob_btts IS NULL THEN NULL ELSE (source.prob_btts >= 0.5) = source.btts_actual END,
    btts_brier_score = CASE WHEN source.prob_btts IS NULL THEN NULL ELSE POWER(source.prob_btts - source.btts_actual::INT, 2) END
FROM source
WHERE performance.id = source.id;

COMMIT;
