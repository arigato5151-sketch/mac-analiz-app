-- Keep pre-match Telegram cards concise regardless of their delivery path.
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
BEGIN
    SELECT match_data.match_date, home_team.name AS home_team_name, away_team.name AS away_team_name
    INTO fixture
    FROM public.matches AS match_data
    JOIN public.teams AS home_team ON home_team.id = match_data.home_team_id
    JOIN public.teams AS away_team ON away_team.id = match_data.away_team_id
    WHERE match_data.id = NEW.match_id;

    SELECT prob_home_win, prob_draw, prob_away_win, prob_over_2_5, prob_btts
    INTO forecast
    FROM public.predictions
    WHERE match_id = NEW.match_id
    ORDER BY predicted_at DESC, id DESC LIMIT 1;

    IF fixture IS NULL OR forecast IS NULL THEN
        RETURN NEW;
    END IF;
    SELECT label, probability INTO outcome_label, outcome_probability
    FROM (VALUES
        ('Ev kazanır'::TEXT, forecast.prob_home_win),
        ('Beraberlik'::TEXT, forecast.prob_draw),
        ('Deplasman kazanır'::TEXT, forecast.prob_away_win)
    ) AS outcomes(label, probability)
    ORDER BY probability DESC LIMIT 1;

    NEW.message_text := format(
        E'⚽ %s — %s\n⏰ %s\n\nTahmin: %s %s%%\nÜst 2.5: %s%% · KG Var: %s%%',
        fixture.home_team_name, fixture.away_team_name,
        TO_CHAR(fixture.match_date AT TIME ZONE 'Europe/Istanbul', 'DD.MM · HH24:MI'),
        outcome_label, ROUND(outcome_probability * 100),
        ROUND(forecast.prob_over_2_5 * 100), ROUND(forecast.prob_btts * 100)
    );
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS simplify_pre_match_telegram_message ON public.pre_match_telegram_queue;
CREATE TRIGGER simplify_pre_match_telegram_message
BEFORE INSERT OR UPDATE OF match_id ON public.pre_match_telegram_queue
FOR EACH ROW EXECUTE FUNCTION public.simplify_pre_match_telegram_message();

COMMIT;
