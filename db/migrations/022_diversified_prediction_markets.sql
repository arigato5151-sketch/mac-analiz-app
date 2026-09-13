-- Persist diversified markets, evaluate them, and include them in automatic Telegram cards.
BEGIN;

ALTER TABLE public.predictions
    ADD COLUMN IF NOT EXISTS market_probabilities JSONB NOT NULL DEFAULT '{}'::JSONB
    CHECK (JSONB_TYPEOF(market_probabilities) = 'object');

ALTER TABLE public.prediction_snapshots
    ADD COLUMN IF NOT EXISTS market_probabilities JSONB NOT NULL DEFAULT '{}'::JSONB
    CHECK (JSONB_TYPEOF(market_probabilities) = 'object');

ALTER TABLE public.shadow_predictions
    ADD COLUMN IF NOT EXISTS market_probabilities JSONB NOT NULL DEFAULT '{}'::JSONB
    CHECK (JSONB_TYPEOF(market_probabilities) = 'object');

ALTER TABLE public.prediction_performance
    ADD COLUMN IF NOT EXISTS market_performance JSONB NOT NULL DEFAULT '{}'::JSONB
    CHECK (JSONB_TYPEOF(market_performance) = 'object');

ALTER TABLE public.shadow_prediction_performance
    ADD COLUMN IF NOT EXISTS market_performance JSONB NOT NULL DEFAULT '{}'::JSONB
    CHECK (JSONB_TYPEOF(market_performance) = 'object');

CREATE OR REPLACE FUNCTION public.format_diversified_prediction_markets(markets JSONB)
RETURNS TEXT
LANGUAGE plpgsql
IMMUTABLE
SET search_path = public, pg_temp
AS $$
DECLARE
    summary TEXT := '';
    double_chance_label TEXT;
    double_chance_probability NUMERIC;
    total_signals TEXT := '';
    team_signals TEXT := '';
    score_signals TEXT;
    probability NUMERIC;
BEGIN
    IF markets IS NULL OR markets = '{}'::JSONB THEN
        RETURN '';
    END IF;

    SELECT entry.key, (entry.value #>> '{}')::NUMERIC
    INTO double_chance_label, double_chance_probability
    FROM JSONB_EACH(markets->'double_chance') AS entry
    ORDER BY (entry.value #>> '{}')::NUMERIC DESC,
             CASE entry.key WHEN '1X' THEN 1 WHEN 'X2' THEN 2 ELSE 3 END
    LIMIT 1;
    IF double_chance_probability >= 0.70 THEN
        summary := summary || format(E'\nÇifte şans: %s %s%%', double_chance_label, ROUND(double_chance_probability * 100));
    END IF;

    probability := COALESCE((markets#>>'{total_goals,over_1_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        total_signals := format('Üst 1.5 %s%%', ROUND(probability * 100));
    END IF;
    probability := COALESCE((markets#>>'{total_goals,under_3_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        total_signals := CONCAT_WS(' · ', NULLIF(total_signals, ''), format('Alt 3.5 %s%%', ROUND(probability * 100)));
    END IF;
    IF total_signals <> '' THEN
        summary := summary || E'\nGol çizgisi: ' || total_signals;
    END IF;

    probability := COALESCE((markets#>>'{team_goals,home_over_0_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        team_signals := format('Ev 0.5 Üst %s%%', ROUND(probability * 100));
    END IF;
    probability := COALESCE((markets#>>'{team_goals,away_over_0_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        team_signals := CONCAT_WS(' · ', NULLIF(team_signals, ''), format('Dep. 0.5 Üst %s%%', ROUND(probability * 100)));
    END IF;
    probability := COALESCE((markets#>>'{team_goals,home_over_1_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        team_signals := CONCAT_WS(' · ', NULLIF(team_signals, ''), format('Ev 1.5 Üst %s%%', ROUND(probability * 100)));
    END IF;
    probability := COALESCE((markets#>>'{team_goals,away_over_1_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        team_signals := CONCAT_WS(' · ', NULLIF(team_signals, ''), format('Dep. 1.5 Üst %s%%', ROUND(probability * 100)));
    END IF;
    IF team_signals <> '' THEN
        summary := summary || E'\nTakım golü: ' || team_signals;
    END IF;

    SELECT STRING_AGG(
        format('%s %s%%', item.value->>'score', ROUND((item.value->>'probability')::NUMERIC * 100)),
        ' · ' ORDER BY item.ordinality
    )
    INTO score_signals
    FROM JSONB_ARRAY_ELEMENTS(COALESCE(markets->'correct_scores', '[]'::JSONB))
        WITH ORDINALITY AS item(value, ordinality)
    WHERE item.ordinality <= 3;
    IF COALESCE(score_signals, '') <> '' THEN
        summary := summary || E'\nOlası skorlar: ' || score_signals;
    END IF;
    RETURN summary;
END;
$$;

CREATE OR REPLACE VIEW public.live_prediction_performance
WITH (security_invoker = false)
AS
WITH ranked AS (
    SELECT
        performance.prediction_id,
        performance.match_id,
        performance.actual_result,
        performance.was_correct,
        performance.brier_score,
        performance.evaluated_at,
        performance.snapshot_id,
        performance.over_2_5_actual,
        performance.over_2_5_was_correct,
        performance.over_2_5_brier_score,
        performance.btts_actual,
        performance.btts_was_correct,
        performance.btts_brier_score,
        performance.market_performance,
        ROW_NUMBER() OVER (
            PARTITION BY performance.match_id
            ORDER BY performance.evaluated_at DESC,
                     prediction.predicted_at DESC,
                     prediction.id DESC
        ) AS row_rank
    FROM public.prediction_performance AS performance
    JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
)
SELECT
    prediction_id,
    match_id,
    actual_result,
    was_correct,
    brier_score,
    evaluated_at,
    snapshot_id,
    over_2_5_actual,
    over_2_5_was_correct,
    over_2_5_brier_score,
    btts_actual,
    btts_was_correct,
    btts_brier_score,
    market_performance
FROM ranked
WHERE row_rank = 1;

CREATE OR REPLACE VIEW public.evaluated_prediction_results
WITH (security_invoker = false)
AS
SELECT
    performance.prediction_id,
    performance.match_id,
    performance.actual_result,
    performance.was_correct,
    performance.brier_score,
    performance.evaluated_at,
    fixture.league_id,
    fixture.match_date,
    fixture.home_score,
    fixture.away_score,
    COALESCE(snapshot.prob_home_win, prediction.prob_home_win) AS prob_home_win,
    COALESCE(snapshot.prob_draw, prediction.prob_draw) AS prob_draw,
    COALESCE(snapshot.prob_away_win, prediction.prob_away_win) AS prob_away_win,
    COALESCE(snapshot.prob_over_2_5, prediction.prob_over_2_5) AS prob_over_2_5,
    COALESCE(snapshot.prob_btts, prediction.prob_btts) AS prob_btts,
    COALESCE(snapshot.model_version, prediction.model_version) AS model_version,
    COALESCE(snapshot.captured_at, prediction.predicted_at) AS predicted_at,
    home.name AS home_team,
    away.name AS away_team,
    league.name AS league_name,
    performance.over_2_5_actual,
    performance.over_2_5_was_correct,
    performance.over_2_5_brier_score,
    performance.btts_actual,
    performance.btts_was_correct,
    performance.btts_brier_score,
    COALESCE(snapshot.market_probabilities, prediction.market_probabilities) AS market_probabilities,
    performance.market_performance
FROM public.live_prediction_performance AS performance
JOIN public.matches AS fixture ON fixture.id = performance.match_id
JOIN public.predictions AS prediction ON prediction.id = performance.prediction_id
LEFT JOIN public.prediction_snapshots AS snapshot ON snapshot.id = performance.snapshot_id
LEFT JOIN public.teams AS home ON home.id = fixture.home_team_id
LEFT JOIN public.teams AS away ON away.id = fixture.away_team_id
LEFT JOIN public.leagues AS league ON league.id = fixture.league_id;

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

    UPDATE public.pre_match_telegram_queue AS queue
    SET delivered_at = NOW(), last_error = NULL
    FROM net._http_response AS response
    WHERE response.id = queue.request_id
      AND response.status_code BETWEEN 200 AND 299
      AND queue.delivered_at IS NULL;

    INSERT INTO public.notification_log (match_id, notification_type, model_version)
    SELECT match_id, 'pre_match_20m', model_version
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
               prediction.prob_over_2_5, prediction.prob_btts, prediction.market_probabilities
        FROM public.matches AS match_data
        JOIN public.teams AS home_team ON home_team.id = match_data.home_team_id
        JOIN public.teams AS away_team ON away_team.id = match_data.away_team_id
        LEFT JOIN public.leagues AS league ON league.id = match_data.league_id
        JOIN LATERAL (
            SELECT * FROM public.predictions WHERE match_id = match_data.id
            ORDER BY predicted_at DESC, id DESC LIMIT 1
        ) AS prediction ON TRUE
        LEFT JOIN public.notification_log AS sent ON sent.match_id = match_data.id
            AND sent.notification_type = 'pre_match_20m'
        WHERE match_data.status = 'scheduled'
          AND match_data.match_date BETWEEN NOW() + INTERVAL '15 minutes' AND NOW() + INTERVAL '25 minutes'
          AND sent.match_id IS NULL
    LOOP
        INSERT INTO public.pre_match_telegram_queue (match_id, model_version, message_text)
        VALUES (
            candidate.id, candidate.model_version,
            format(E'⚽ %s — %s\n📍 %s · %s\n──────────────\n🤖 Model olasılıkları\n1X2  1 %s%% · X %s%% · 2 %s%%%s%s%s\n\nİstatistiksel olasılıktır; kesin sonuç değildir.',
                candidate.home_team_name, candidate.away_team_name, COALESCE(candidate.league_name, 'Lig bilgisi yok'),
                TO_CHAR(candidate.match_date AT TIME ZONE 'Europe/Istanbul', 'DD.MM HH24:MI'),
                ROUND(candidate.prob_home_win * 100), ROUND(candidate.prob_draw * 100), ROUND(candidate.prob_away_win * 100),
                CASE WHEN candidate.prob_over_2_5 IS NULL THEN '' ELSE format(E'\n2.5  Üst %s%% · Alt %s%%', ROUND(candidate.prob_over_2_5 * 100), ROUND((1 - candidate.prob_over_2_5) * 100)) END,
                CASE WHEN candidate.prob_btts IS NULL THEN '' ELSE format(E'\nKG   Var %s%% · Yok %s%%', ROUND(candidate.prob_btts * 100), ROUND((1 - candidate.prob_btts) * 100)) END,
                public.format_diversified_prediction_markets(candidate.market_probabilities)
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

GRANT SELECT ON public.live_prediction_performance,
    public.evaluated_prediction_results
TO anon, service_role;
GRANT EXECUTE ON FUNCTION public.format_diversified_prediction_markets(JSONB) TO service_role;

COMMIT;
