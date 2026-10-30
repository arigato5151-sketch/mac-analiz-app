-- Persist display-ready pre-match analysis for fast Streamlit match cards.
BEGIN;

CREATE TABLE public.match_analysis_cache (
    match_id INTEGER PRIMARY KEY REFERENCES public.matches(id) ON DELETE CASCADE,
    league_id INTEGER NOT NULL,
    home_team_id INTEGER NOT NULL,
    away_team_id INTEGER NOT NULL,
    match_date TIMESTAMPTZ NOT NULL,
    model_version TEXT NOT NULL,
    calculated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    analysis_json JSONB NOT NULL CHECK (jsonb_typeof(analysis_json) = 'object')
);

CREATE INDEX idx_match_analysis_cache_date
    ON public.match_analysis_cache(match_date, match_id);

ALTER TABLE public.match_analysis_cache ENABLE ROW LEVEL SECURITY;

CREATE POLICY match_analysis_cache_public_read
    ON public.match_analysis_cache
    FOR SELECT TO anon, authenticated
    USING (true);

REVOKE ALL ON TABLE public.match_analysis_cache FROM PUBLIC;
REVOKE ALL ON TABLE public.match_analysis_cache FROM anon, authenticated;
GRANT SELECT ON TABLE public.match_analysis_cache TO anon, authenticated;
GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.match_analysis_cache TO service_role;

COMMIT;
