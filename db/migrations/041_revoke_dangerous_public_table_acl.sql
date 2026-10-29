-- Remove table privileges that are not valid API operations and are never needed for public roles.
BEGIN;

REVOKE TRUNCATE, REFERENCES, TRIGGER ON TABLE
    public.schema_migrations,
    public.result_notification_queue,
    public.notification_log,
    public.model_candidates,
    public.shadow_predictions,
    public.shadow_prediction_performance,
    public.personal_match_workspace,
    public.personal_match_decisions,
    public.bookmaker_odds_snapshots
FROM anon, authenticated;

-- PostgreSQL added MAINTAIN in version 17. Keep it out of static SQL so this
-- migration also parses on older supported servers.
DO $$
BEGIN
    IF current_setting('server_version_num')::integer >= 170000 THEN
        EXECUTE 'REVOKE MAINTAIN ON TABLE
            public.schema_migrations,
            public.result_notification_queue,
            public.notification_log,
            public.model_candidates,
            public.shadow_predictions,
            public.shadow_prediction_performance,
            public.personal_match_workspace,
            public.personal_match_decisions,
            public.bookmaker_odds_snapshots
        FROM anon, authenticated';
    END IF;
END;
$$;

-- Seed-free post-apply verification; run separately after applying this migration.
-- role_table_grants check for TRUNCATE, REFERENCES, and TRIGGER:
-- SELECT grantee, table_name, privilege_type
-- FROM information_schema.role_table_grants
-- WHERE table_schema = 'public'
--   AND grantee IN ('anon', 'authenticated')
--   AND table_name IN (
--       'schema_migrations', 'result_notification_queue', 'notification_log',
--       'model_candidates', 'shadow_predictions',
--       'shadow_prediction_performance', 'personal_match_workspace',
--       'personal_match_decisions', 'bookmaker_odds_snapshots'
--   )
--   AND privilege_type IN ('TRUNCATE', 'REFERENCES', 'TRIGGER')
-- ORDER BY grantee, table_name, privilege_type;
--
-- PostgreSQL 17+ MAINTAIN effective-privilege check (expected: zero rows):
-- SELECT r.grantee, t.table_name, 'MAINTAIN' AS privilege_type
-- FROM (VALUES ('anon'), ('authenticated')) AS r(grantee)
-- CROSS JOIN (VALUES
--     ('schema_migrations'), ('result_notification_queue'), ('notification_log'),
--     ('model_candidates'), ('shadow_predictions'),
--     ('shadow_prediction_performance'), ('personal_match_workspace'),
--     ('personal_match_decisions'), ('bookmaker_odds_snapshots')
-- ) AS t(table_name)
-- WHERE current_setting('server_version_num')::integer >= 170000
--   AND has_table_privilege(r.grantee::name, format('public.%I', t.table_name), 'MAINTAIN')
-- ORDER BY r.grantee, t.table_name;

COMMIT;
