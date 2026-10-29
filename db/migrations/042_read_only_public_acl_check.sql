-- Expose a restricted, read-only check of the public-role table ACLs for CI.
BEGIN;

CREATE OR REPLACE FUNCTION public.verify_public_table_acl()
RETURNS TABLE (grantee TEXT, table_name TEXT, privilege_type TEXT)
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $function$
DECLARE
    target_role TEXT;
    target_table TEXT;
    target_privilege TEXT;
BEGIN
    FOREACH target_role IN ARRAY ARRAY['anon', 'authenticated'] LOOP
        FOREACH target_table IN ARRAY ARRAY[
            'schema_migrations',
            'result_notification_queue',
            'notification_log',
            'model_candidates',
            'shadow_predictions',
            'shadow_prediction_performance',
            'personal_match_workspace',
            'personal_match_decisions',
            'bookmaker_odds_snapshots'
        ] LOOP
            FOREACH target_privilege IN ARRAY ARRAY[
                'TRUNCATE', 'REFERENCES', 'TRIGGER'
            ] LOOP
                IF has_table_privilege(
                    target_role::name,
                    format('public.%I', target_table),
                    target_privilege
                ) THEN
                    grantee := target_role;
                    table_name := target_table;
                    privilege_type := target_privilege;
                    RETURN NEXT;
                END IF;
            END LOOP;

            IF current_setting('server_version_num')::integer >= 170000 THEN
                IF has_table_privilege(
                    target_role::name,
                    format('public.%I', target_table),
                    'MAINTAIN'
                ) THEN
                    grantee := target_role;
                    table_name := target_table;
                    privilege_type := 'MAINTAIN';
                    RETURN NEXT;
                END IF;
            END IF;
        END LOOP;
    END LOOP;
END;
$function$;

REVOKE ALL ON FUNCTION public.verify_public_table_acl()
FROM PUBLIC, anon, authenticated;
GRANT EXECUTE ON FUNCTION public.verify_public_table_acl() TO service_role;

COMMIT;
