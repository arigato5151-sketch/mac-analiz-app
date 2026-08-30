-- Confirmed starting elevens are separate from injury/availability context.
BEGIN;

CREATE TABLE IF NOT EXISTS public.fixture_lineups (
    match_id INTEGER NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    team_id INTEGER NOT NULL REFERENCES public.teams(id) ON DELETE CASCADE,
    formation TEXT,
    coach_name TEXT,
    starters JSONB NOT NULL DEFAULT '[]'::JSONB,
    substitutes JSONB NOT NULL DEFAULT '[]'::JSONB,
    confirmed_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    PRIMARY KEY (match_id, team_id)
);

CREATE INDEX IF NOT EXISTS idx_fixture_lineups_match ON public.fixture_lineups(match_id);
GRANT SELECT, INSERT, UPDATE ON public.fixture_lineups TO service_role;
REVOKE ALL ON public.fixture_lineups FROM anon, authenticated;
GRANT SELECT ON public.fixture_lineups TO anon;
ALTER TABLE public.fixture_lineups ENABLE ROW LEVEL SECURITY;
DROP POLICY IF EXISTS "public_read_fixture_lineups" ON public.fixture_lineups;
CREATE POLICY "public_read_fixture_lineups" ON public.fixture_lineups FOR SELECT TO anon USING (true);

COMMIT;
