-- Store provider corner statistics for causal corner forecasts.
BEGIN;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS home_corners INTEGER CHECK (home_corners IS NULL OR home_corners >= 0),
    ADD COLUMN IF NOT EXISTS away_corners INTEGER CHECK (away_corners IS NULL OR away_corners >= 0),
    ADD COLUMN IF NOT EXISTS corner_stats_checked_at TIMESTAMPTZ;

CREATE INDEX IF NOT EXISTS idx_matches_finished_team_date_corners
    ON public.matches(status, match_date, home_team_id, away_team_id)
    WHERE home_corners IS NOT NULL AND away_corners IS NOT NULL;

COMMIT;
