-- Least-privilege browser access for Streamlit.
-- Backend jobs continue using service_role, which bypasses RLS.
BEGIN;

REVOKE ALL ON TABLE leagues, teams, matches, match_stats, team_form,
    player_availability, team_availability_status, predictions, prediction_performance
FROM anon, authenticated;

GRANT SELECT ON TABLE leagues, teams, matches, team_form, player_availability,
    team_availability_status, predictions, prediction_performance TO anon;

ALTER TABLE leagues ENABLE ROW LEVEL SECURITY;
ALTER TABLE teams ENABLE ROW LEVEL SECURITY;
ALTER TABLE matches ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_form ENABLE ROW LEVEL SECURITY;
ALTER TABLE predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE prediction_performance ENABLE ROW LEVEL SECURITY;
ALTER TABLE player_availability ENABLE ROW LEVEL SECURITY;
ALTER TABLE team_availability_status ENABLE ROW LEVEL SECURITY;

DROP POLICY IF EXISTS "public_read_leagues" ON leagues;
CREATE POLICY "public_read_leagues" ON leagues FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_teams" ON teams;
CREATE POLICY "public_read_teams" ON teams FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_matches" ON matches;
CREATE POLICY "public_read_matches" ON matches FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_team_form" ON team_form;
CREATE POLICY "public_read_team_form" ON team_form FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_predictions" ON predictions;
CREATE POLICY "public_read_predictions" ON predictions FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_prediction_performance" ON prediction_performance;
CREATE POLICY "public_read_prediction_performance" ON prediction_performance FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_player_availability" ON player_availability;
CREATE POLICY "public_read_player_availability" ON player_availability FOR SELECT TO anon USING (true);
DROP POLICY IF EXISTS "public_read_team_availability_status" ON team_availability_status;
CREATE POLICY "public_read_team_availability_status" ON team_availability_status FOR SELECT TO anon USING (true);

COMMIT;
