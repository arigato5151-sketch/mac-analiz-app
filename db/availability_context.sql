-- Persist a per-team refresh marker and expose only the public squad context.
BEGIN;

CREATE TABLE IF NOT EXISTS public.team_availability_status (
    team_id INTEGER PRIMARY KEY REFERENCES public.teams(id) ON DELETE CASCADE,
    refreshed_at TIMESTAMPTZ NOT NULL,
    available_count INTEGER NOT NULL DEFAULT 0 CHECK (available_count >= 0)
);

GRANT SELECT, INSERT, UPDATE, DELETE ON TABLE public.team_availability_status TO service_role;
REVOKE ALL ON TABLE public.player_availability, public.team_availability_status
FROM anon, authenticated;
GRANT SELECT ON TABLE public.player_availability, public.team_availability_status TO anon;

ALTER TABLE public.player_availability ENABLE ROW LEVEL SECURITY;
ALTER TABLE public.team_availability_status ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "public_read_player_availability" ON public.player_availability;
CREATE POLICY "public_read_player_availability" ON public.player_availability
FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_team_availability_status" ON public.team_availability_status;
CREATE POLICY "public_read_team_availability_status" ON public.team_availability_status
FOR SELECT TO anon USING (true);

COMMIT;
