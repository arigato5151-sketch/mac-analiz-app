-- Store provider-supplied expected assists alongside existing expected goals.
BEGIN;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS home_xa NUMERIC CHECK (home_xa IS NULL OR home_xa >= 0),
    ADD COLUMN IF NOT EXISTS away_xa NUMERIC CHECK (away_xa IS NULL OR away_xa >= 0),
    ADD COLUMN IF NOT EXISTS expected_metrics_checked_at TIMESTAMPTZ;

COMMIT;
