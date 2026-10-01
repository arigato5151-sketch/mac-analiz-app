-- Keep current availability snapshots compatible with fixture-scoped ingestion.
BEGIN;

ALTER TABLE public.team_availability_status
    ADD COLUMN IF NOT EXISTS match_id INTEGER
    REFERENCES public.matches(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_team_availability_status_match
    ON public.team_availability_status(match_id);

COMMIT;
