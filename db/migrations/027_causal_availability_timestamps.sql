-- Preserve availability provenance and fixture scope for causal training.
BEGIN;

ALTER TABLE public.team_availability_history
    ADD COLUMN IF NOT EXISTS match_id INTEGER REFERENCES public.matches(id) ON DELETE CASCADE,
    ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;

UPDATE public.team_availability_history
SET ingested_at = refreshed_at
WHERE ingested_at IS NULL;

ALTER TABLE public.team_availability_history
    ALTER COLUMN ingested_at SET NOT NULL;

CREATE INDEX IF NOT EXISTS idx_availability_history_fixture_time
    ON public.team_availability_history(match_id, team_id, ingested_at DESC);

COMMIT;
