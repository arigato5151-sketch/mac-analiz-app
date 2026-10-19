-- Store provider halftime scores for separate first and second half forecasts.
BEGIN;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS home_halftime_score INTEGER CHECK (home_halftime_score IS NULL OR home_halftime_score >= 0),
    ADD COLUMN IF NOT EXISTS away_halftime_score INTEGER CHECK (away_halftime_score IS NULL OR away_halftime_score >= 0);

CREATE INDEX IF NOT EXISTS idx_matches_finished_halftime_history
    ON public.matches(league_id, match_date DESC)
    WHERE status = 'finished' AND home_halftime_score IS NOT NULL AND away_halftime_score IS NOT NULL;

COMMIT;
