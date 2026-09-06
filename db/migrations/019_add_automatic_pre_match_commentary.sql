-- Add a deterministic commentary fallback to Supabase-scheduled Telegram cards.
-- The GitHub path still uses Gemini; this keeps database-delivered alerts useful
-- when GitHub's scheduled workflows are delayed.
BEGIN;

CREATE OR REPLACE FUNCTION public.simplify_pre_match_telegram_message()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
DECLARE
    fixture RECORD;
    forecast RECORD;
    outcome_label TEXT;
    outcome_probability NUMERIC;
    outcome_comment TEXT;
    goals_comment TEXT;
    btts_comment TEXT;
BEGIN
    SELECT
        match_data.match_date,
        home_team.name AS home_team_name,
        away_team.name AS away_team_name
    INTO fixture
    FROM public.matches AS match_data
    JOIN public.teams AS home_team ON home_team.id = match_data.home_team_id
    JOIN public.teams AS away_team ON away_team.id = match_data.away_team_id
    WHERE match_data.id = NEW.match_id;

    SELECT prob_home_win, prob_draw, prob_away_win, prob_over_2_5, prob_btts
    INTO forecast
    FROM public.predictions
    WHERE match_id = NEW.match_id
    ORDER BY predicted_at DESC, id DESC
    LIMIT 1;

    IF fixture IS NULL OR forecast IS NULL THEN
        RETURN NEW;
    END IF;

    SELECT label, probability
    INTO outcome_label, outcome_probability
    FROM (VALUES
        ('Ev sahibini'::TEXT, forecast.prob_home_win),
        ('Beraberliği'::TEXT, forecast.prob_draw),
        ('Deplasmanı'::TEXT, forecast.prob_away_win)
    ) AS outcomes(label, probability)
    ORDER BY probability DESC
    LIMIT 1;

    outcome_comment := CASE
        WHEN outcome_probability >= 0.65 THEN
            format('Model %s güçlü biçimde öne çıkarıyor (%%%s).', outcome_label, ROUND(outcome_probability * 100))
        WHEN outcome_probability >= 0.50 THEN
            format('Model %s hafif favori görüyor (%%%s); sonuç güveni sınırlı.', outcome_label, ROUND(outcome_probability * 100))
        ELSE
            '1-X-2 olasılıkları birbirine yakın; belirgin bir favori görünmüyor.'
    END;

    goals_comment := CASE
        WHEN forecast.prob_over_2_5 >= 0.60 THEN
            format('Üst 2.5 olasılığı %%%s ile gollü senaryoyu destekliyor.', ROUND(forecast.prob_over_2_5 * 100))
        WHEN forecast.prob_over_2_5 <= 0.40 THEN
            format('Üst 2.5 olasılığı yalnızca %%%s; düşük skorlu senaryo daha güçlü.', ROUND(forecast.prob_over_2_5 * 100))
        ELSE
            format('Gol beklentisi dengeli; Üst 2.5 olasılığı %%%s.', ROUND(forecast.prob_over_2_5 * 100))
    END;

    btts_comment := CASE
        WHEN forecast.prob_btts >= 0.60 THEN
            format('KG Var ihtimali %%%s ile iki takımın da gol bulabileceğine işaret ediyor.', ROUND(forecast.prob_btts * 100))
        WHEN forecast.prob_btts <= 0.40 THEN
            format('KG Var ihtimali %%%s; takımlardan birinin gol bulamaması daha olası.', ROUND(forecast.prob_btts * 100))
        ELSE
            format('KG Var tarafı da %%%s ile kararsız bölgede.', ROUND(forecast.prob_btts * 100))
    END;

    NEW.message_text := format(
        E'⚽ %s — %s\n⏰ %s\n\nTahmin: %s %s%%\nÜst 2.5: %s%% · KG Var: %s%%\n\n🧠 Maç yorumu\n%s %s %s',
        fixture.home_team_name,
        fixture.away_team_name,
        TO_CHAR(fixture.match_date AT TIME ZONE 'Europe/Istanbul', 'DD.MM · HH24:MI'),
        CASE outcome_label
            WHEN 'Ev sahibini' THEN 'Ev kazanır'
            WHEN 'Beraberliği' THEN 'Beraberlik'
            ELSE 'Deplasman kazanır'
        END,
        ROUND(outcome_probability * 100),
        ROUND(forecast.prob_over_2_5 * 100),
        ROUND(forecast.prob_btts * 100),
        outcome_comment,
        goals_comment,
        btts_comment
    );
    RETURN NEW;
END;
$$;

-- Refresh only queued, unsent rows so retries also use the corrected template.
UPDATE public.pre_match_telegram_queue
SET match_id = match_id
WHERE delivered_at IS NULL;

COMMIT;
