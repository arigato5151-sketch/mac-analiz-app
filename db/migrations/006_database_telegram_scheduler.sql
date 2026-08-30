-- Deliver compact pre-match Telegram alerts from Supabase, independent of GitHub cron latency.
-- Required Vault secret names (their values are never stored in this repository):
--   telegram_bot_token, telegram_chat_id
BEGIN;

CREATE OR REPLACE FUNCTION public.dispatch_due_telegram_pre_match_alerts()
RETURNS INTEGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, vault, net, pg_temp
AS $$
DECLARE
    telegram_token TEXT;
    telegram_chat_id TEXT;
    candidate RECORD;
    claimed_notification_id BIGINT;
    compact_message TEXT;
    dispatched_count INTEGER := 0;
BEGIN
    SELECT decrypted_secret INTO telegram_token
    FROM vault.decrypted_secrets
    WHERE name = 'telegram_bot_token';

    SELECT decrypted_secret INTO telegram_chat_id
    FROM vault.decrypted_secrets
    WHERE name = 'telegram_chat_id';

    IF COALESCE(telegram_token, '') = '' OR COALESCE(telegram_chat_id, '') = '' THEN
        RAISE EXCEPTION 'Telegram Vault secrets are not configured';
    END IF;

    FOR candidate IN
        SELECT
            match_data.id AS match_id,
            match_data.match_date,
            league.name AS league_name,
            home_team.name AS home_team_name,
            away_team.name AS away_team_name,
            prediction.model_version,
            prediction.prob_home_win,
            prediction.prob_draw,
            prediction.prob_away_win,
            prediction.prob_over_2_5,
            prediction.prob_btts
        FROM public.matches AS match_data
        INNER JOIN public.teams AS home_team ON home_team.id = match_data.home_team_id
        INNER JOIN public.teams AS away_team ON away_team.id = match_data.away_team_id
        LEFT JOIN public.leagues AS league ON league.id = match_data.league_id
        INNER JOIN LATERAL (
            SELECT latest_prediction.*
            FROM public.predictions AS latest_prediction
            WHERE latest_prediction.match_id = match_data.id
            ORDER BY latest_prediction.predicted_at DESC, latest_prediction.id DESC
            LIMIT 1
        ) AS prediction ON TRUE
        WHERE match_data.status = 'scheduled'
          AND match_data.match_date >= NOW() + INTERVAL '10 minutes'
          AND match_data.match_date <= NOW() + INTERVAL '90 minutes'
        ORDER BY match_data.match_date ASC
    LOOP
        -- The unique log row is an atomic claim, preventing concurrent cron runs from double-sending.
        INSERT INTO public.notification_log (match_id, notification_type, model_version)
        VALUES (candidate.match_id, 'pre_match_60m', candidate.model_version)
        ON CONFLICT (match_id, notification_type) DO NOTHING
        RETURNING id INTO claimed_notification_id;

        IF claimed_notification_id IS NULL THEN
            CONTINUE;
        END IF;

        compact_message := format(
            E'⚽ %s — %s\n📍 %s · %s\n──────────────\n🤖 Model olasılıkları\n1X2  1 %s%% · X %s%% · 2 %s%%%s%s\n\nİstatistiksel olasılıktır; kesin sonuç değildir.',
            candidate.home_team_name,
            candidate.away_team_name,
            COALESCE(candidate.league_name, 'Lig bilgisi yok'),
            TO_CHAR(candidate.match_date AT TIME ZONE 'Europe/Istanbul', 'DD.MM HH24:MI'),
            ROUND(candidate.prob_home_win * 100),
            ROUND(candidate.prob_draw * 100),
            ROUND(candidate.prob_away_win * 100),
            CASE WHEN candidate.prob_over_2_5 IS NULL THEN ''
                 ELSE format(E'\n2.5  Üst %s%% · Alt %s%%', ROUND(candidate.prob_over_2_5 * 100), ROUND((1 - candidate.prob_over_2_5) * 100)) END,
            CASE WHEN candidate.prob_btts IS NULL THEN ''
                 ELSE format(E'\nKG   Var %s%% · Yok %s%%', ROUND(candidate.prob_btts * 100), ROUND((1 - candidate.prob_btts) * 100)) END
        );

        PERFORM net.http_post(
            url := 'https://api.telegram.org/bot' || telegram_token || '/sendMessage',
            headers := '{"Content-Type": "application/json"}'::JSONB,
            body := JSONB_BUILD_OBJECT(
                'chat_id', telegram_chat_id,
                'text', compact_message,
                'disable_web_page_preview', TRUE
            ),
            timeout_milliseconds := 5000
        );
        dispatched_count := dispatched_count + 1;
    END LOOP;

    RETURN dispatched_count;
END;
$$;

REVOKE ALL ON FUNCTION public.dispatch_due_telegram_pre_match_alerts() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.dispatch_due_telegram_pre_match_alerts() TO service_role;

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

    PERFORM cron.schedule(
        'dispatch-telegram-pre-match-alerts',
        '*/5 * * * *',
        'SELECT public.dispatch_due_telegram_pre_match_alerts();'
    );
END;
$$;

COMMIT;
