-- Queue pre-match delivery until Telegram confirms HTTP success.
BEGIN;

CREATE TABLE IF NOT EXISTS public.pre_match_telegram_queue (
    match_id INTEGER PRIMARY KEY REFERENCES public.matches(id) ON DELETE CASCADE,
    model_version TEXT NOT NULL,
    message_text TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    request_id BIGINT,
    delivered_at TIMESTAMPTZ,
    last_error TEXT
);
CREATE INDEX IF NOT EXISTS idx_pre_match_telegram_queue_pending
    ON public.pre_match_telegram_queue(next_attempt_at)
    WHERE delivered_at IS NULL;
GRANT SELECT, INSERT, UPDATE, DELETE ON public.pre_match_telegram_queue TO service_role;
REVOKE ALL ON public.pre_match_telegram_queue FROM anon, authenticated;

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
    queued_count INTEGER := 0;
BEGIN
    SELECT decrypted_secret INTO telegram_token FROM vault.decrypted_secrets WHERE name = 'telegram_bot_token';
    SELECT decrypted_secret INTO telegram_chat_id FROM vault.decrypted_secrets WHERE name = 'telegram_chat_id';
    IF COALESCE(telegram_token, '') = '' OR COALESCE(telegram_chat_id, '') = '' THEN
        RAISE EXCEPTION 'Telegram Vault secrets are not configured';
    END IF;

    -- Confirm only successful HTTP deliveries; timeouts stay retryable.
    UPDATE public.pre_match_telegram_queue AS queue
    SET delivered_at = NOW(), last_error = NULL
    FROM net._http_response AS response
    WHERE response.id = queue.request_id
      AND response.status_code BETWEEN 200 AND 299
      AND queue.delivered_at IS NULL;

    INSERT INTO public.notification_log (match_id, notification_type, model_version)
    SELECT match_id, 'pre_match_60m', model_version
    FROM public.pre_match_telegram_queue
    WHERE delivered_at IS NOT NULL
    ON CONFLICT (match_id, notification_type) DO NOTHING;

    UPDATE public.pre_match_telegram_queue AS queue
    SET request_id = NULL,
        next_attempt_at = NOW() + INTERVAL '5 minutes',
        last_error = CASE WHEN response.timed_out THEN 'timeout' ELSE 'telegram_http_error' END
    FROM net._http_response AS response
    WHERE response.id = queue.request_id
      AND queue.delivered_at IS NULL
      AND (response.timed_out OR response.status_code >= 400)
      AND queue.attempts < 3;

    FOR candidate IN
        SELECT match_data.id, prediction.model_version, match_data.match_date, league.name AS league_name,
               home_team.name AS home_team_name, away_team.name AS away_team_name,
               prediction.prob_home_win, prediction.prob_draw, prediction.prob_away_win,
               prediction.prob_over_2_5, prediction.prob_btts
        FROM public.matches AS match_data
        JOIN public.teams AS home_team ON home_team.id = match_data.home_team_id
        JOIN public.teams AS away_team ON away_team.id = match_data.away_team_id
        LEFT JOIN public.leagues AS league ON league.id = match_data.league_id
        JOIN LATERAL (
            SELECT * FROM public.predictions WHERE match_id = match_data.id
            ORDER BY predicted_at DESC, id DESC LIMIT 1
        ) AS prediction ON TRUE
        LEFT JOIN public.notification_log AS sent ON sent.match_id = match_data.id
            AND sent.notification_type = 'pre_match_60m'
        WHERE match_data.status = 'scheduled'
          AND match_data.match_date BETWEEN NOW() + INTERVAL '15 minutes' AND NOW() + INTERVAL '25 minutes'
          AND sent.match_id IS NULL
    LOOP
        INSERT INTO public.pre_match_telegram_queue (match_id, model_version, message_text)
        VALUES (
            candidate.id, candidate.model_version,
            format(E'⚽ %s — %s\n📍 %s · %s\n──────────────\n🤖 Model olasılıkları\n1X2  1 %s%% · X %s%% · 2 %s%%%s%s\n\nİstatistiksel olasılıktır; kesin sonuç değildir.',
                candidate.home_team_name, candidate.away_team_name, COALESCE(candidate.league_name, 'Lig bilgisi yok'),
                TO_CHAR(candidate.match_date AT TIME ZONE 'Europe/Istanbul', 'DD.MM HH24:MI'),
                ROUND(candidate.prob_home_win * 100), ROUND(candidate.prob_draw * 100), ROUND(candidate.prob_away_win * 100),
                CASE WHEN candidate.prob_over_2_5 IS NULL THEN '' ELSE format(E'\n2.5  Üst %s%% · Alt %s%%', ROUND(candidate.prob_over_2_5 * 100), ROUND((1 - candidate.prob_over_2_5) * 100)) END,
                CASE WHEN candidate.prob_btts IS NULL THEN '' ELSE format(E'\nKG   Var %s%% · Yok %s%%', ROUND(candidate.prob_btts * 100), ROUND((1 - candidate.prob_btts) * 100)) END
            )
        ) ON CONFLICT (match_id) DO NOTHING;
    END LOOP;

    UPDATE public.pre_match_telegram_queue
    SET request_id = net.http_post(
            url := 'https://api.telegram.org/bot' || telegram_token || '/sendMessage',
            headers := '{"Content-Type":"application/json"}'::JSONB,
            body := JSONB_BUILD_OBJECT('chat_id', telegram_chat_id, 'text', message_text, 'disable_web_page_preview', TRUE),
            timeout_milliseconds := 20000
        ),
        attempts = attempts + 1,
        next_attempt_at = NOW() + INTERVAL '10 minutes'
    WHERE delivered_at IS NULL AND request_id IS NULL AND next_attempt_at <= NOW() AND attempts < 3;

    GET DIAGNOSTICS queued_count = ROW_COUNT;
    RETURN queued_count;
END;
$$;

COMMIT;
