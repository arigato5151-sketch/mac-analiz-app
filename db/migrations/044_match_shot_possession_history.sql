-- Store observed shot and possession statistics for recent-form analysis.
BEGIN;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS home_shots INTEGER CHECK (home_shots IS NULL OR home_shots >= 0),
    ADD COLUMN IF NOT EXISTS away_shots INTEGER CHECK (away_shots IS NULL OR away_shots >= 0),
    ADD COLUMN IF NOT EXISTS home_shots_on_target INTEGER CHECK (home_shots_on_target IS NULL OR home_shots_on_target >= 0),
    ADD COLUMN IF NOT EXISTS away_shots_on_target INTEGER CHECK (away_shots_on_target IS NULL OR away_shots_on_target >= 0),
    ADD COLUMN IF NOT EXISTS home_possession NUMERIC(5, 2) CHECK (home_possession IS NULL OR home_possession BETWEEN 0 AND 100),
    ADD COLUMN IF NOT EXISTS away_possession NUMERIC(5, 2) CHECK (away_possession IS NULL OR away_possession BETWEEN 0 AND 100);

CREATE INDEX IF NOT EXISTS idx_matches_finished_team_date_shots
    ON public.matches(status, match_date, home_team_id, away_team_id)
    WHERE home_shots IS NOT NULL AND away_shots IS NOT NULL;

COMMIT;
