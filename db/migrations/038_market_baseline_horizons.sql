-- Preserve point-in-time bookmaker probabilities for paired horizon evaluation.
BEGIN;

ALTER TABLE public.prediction_horizon_snapshots
    ADD COLUMN IF NOT EXISTS market_implied_probabilities JSONB NOT NULL DEFAULT '{}'::JSONB;

COMMIT;
