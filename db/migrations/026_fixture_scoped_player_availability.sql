-- Bind player availability to the fixture returned by API-Football.
BEGIN;

ALTER TABLE public.player_availability
    ADD COLUMN IF NOT EXISTS match_id INTEGER
    REFERENCES public.matches(id) ON DELETE CASCADE;

CREATE INDEX IF NOT EXISTS idx_player_availability_match
    ON public.player_availability(match_id);

COMMIT;
