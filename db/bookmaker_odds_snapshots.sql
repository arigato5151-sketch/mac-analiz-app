BEGIN;

CREATE TABLE IF NOT EXISTS public.bookmaker_odds_snapshots (
    prediction_id BIGINT PRIMARY KEY REFERENCES public.predictions(id) ON DELETE CASCADE,
    match_id INTEGER NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    bookmaker TEXT NOT NULL,
    source_updated_at TIMESTAMPTZ,
    odds JSONB NOT NULL,
    captured_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_bookmaker_odds_snapshots_match
ON public.bookmaker_odds_snapshots(match_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON public.bookmaker_odds_snapshots TO service_role;

COMMIT;
