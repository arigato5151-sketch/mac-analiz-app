-- Auditable metrics for the total-goals and both-teams-to-score markets.
BEGIN;

ALTER TABLE public.prediction_performance
    ADD COLUMN IF NOT EXISTS over_2_5_actual BOOLEAN,
    ADD COLUMN IF NOT EXISTS over_2_5_was_correct BOOLEAN,
    ADD COLUMN IF NOT EXISTS over_2_5_brier_score NUMERIC CHECK (over_2_5_brier_score IS NULL OR over_2_5_brier_score BETWEEN 0 AND 1),
    ADD COLUMN IF NOT EXISTS btts_actual BOOLEAN,
    ADD COLUMN IF NOT EXISTS btts_was_correct BOOLEAN,
    ADD COLUMN IF NOT EXISTS btts_brier_score NUMERIC CHECK (btts_brier_score IS NULL OR btts_brier_score BETWEEN 0 AND 1);

COMMIT;
