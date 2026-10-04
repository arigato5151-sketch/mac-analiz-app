BEGIN;

-- Keep the scheduled entry point callable, but stop database-side match alerts.
CREATE OR REPLACE FUNCTION public.dispatch_due_telegram_pre_match_alerts()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    RETURN 0;
END;
$$;

DO $$
DECLARE
    existing_job_id BIGINT;
BEGIN
    SELECT jobid INTO existing_job_id
    FROM cron.job
    WHERE jobname = 'dispatch-telegram-pre-match-alerts';

    IF existing_job_id IS NOT NULL THEN
        PERFORM cron.unschedule(existing_job_id);
    END IF;
END;
$$;

COMMIT;
