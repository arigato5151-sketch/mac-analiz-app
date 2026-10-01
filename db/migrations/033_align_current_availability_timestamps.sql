-- Align current availability tables with the causal timestamps written by the pipeline.
BEGIN;

ALTER TABLE public.player_availability
    ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ;

ALTER TABLE public.team_availability_status
    ADD COLUMN IF NOT EXISTS observed_at TIMESTAMPTZ,
    ADD COLUMN IF NOT EXISTS ingested_at TIMESTAMPTZ;

UPDATE public.team_availability_status
SET ingested_at = refreshed_at
WHERE ingested_at IS NULL;

ALTER TABLE public.team_availability_status
    ALTER COLUMN ingested_at SET NOT NULL;

COMMIT;
