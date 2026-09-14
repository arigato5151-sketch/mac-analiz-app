-- Show every confident diversified market on its own Telegram line.
BEGIN;

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
    probability NUMERIC;
    score_item RECORD;
BEGIN
    IF markets IS NULL OR markets = '{}'::JSONB THEN
        RETURN '';
    END IF;

    SELECT entry.key, (entry.value #>> '{}')::NUMERIC
    INTO double_chance_label, double_chance_probability
    FROM JSONB_EACH(COALESCE(markets->'double_chance', '{}'::JSONB)) AS entry
    ORDER BY (entry.value #>> '{}')::NUMERIC DESC,
             CASE entry.key WHEN '1X' THEN 1 WHEN 'X2' THEN 2 ELSE 3 END
    LIMIT 1;
    IF double_chance_probability >= 0.70 THEN
        summary := summary || format(
            E'\nÇifte şans: %s %s%%',
            double_chance_label,
            ROUND(double_chance_probability * 100)
        );
    END IF;

    probability := COALESCE((markets#>>'{total_goals,over_1_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        summary := summary || format(E'\nÜst 1.5: %s%%', ROUND(probability * 100));
    END IF;

    probability := COALESCE((markets#>>'{total_goals,under_3_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        summary := summary || format(E'\nAlt 3.5: %s%%', ROUND(probability * 100));
    END IF;

    probability := COALESCE((markets#>>'{team_goals,home_over_0_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        summary := summary || format(E'\nEv 0.5 Üst: %s%%', ROUND(probability * 100));
    END IF;

    probability := COALESCE((markets#>>'{team_goals,away_over_0_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        summary := summary || format(E'\nDep. 0.5 Üst: %s%%', ROUND(probability * 100));
    END IF;

    probability := COALESCE((markets#>>'{team_goals,home_over_1_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        summary := summary || format(E'\nEv 1.5 Üst: %s%%', ROUND(probability * 100));
    END IF;

    probability := COALESCE((markets#>>'{team_goals,away_over_1_5}')::NUMERIC, 0);
    IF probability >= 0.65 THEN
        summary := summary || format(E'\nDep. 1.5 Üst: %s%%', ROUND(probability * 100));
    END IF;

    FOR score_item IN
        SELECT item.value, item.ordinality
        FROM JSONB_ARRAY_ELEMENTS(COALESCE(markets->'correct_scores', '[]'::JSONB))
            WITH ORDINALITY AS item(value, ordinality)
        WHERE item.ordinality <= 3
        ORDER BY item.ordinality
    LOOP
        summary := summary || format(
            E'\nSkor %s: %s %s%%',
            score_item.ordinality,
            score_item.value->>'score',
            ROUND((score_item.value->>'probability')::NUMERIC * 100)
        );
    END LOOP;
    IF summary = '' THEN
        RETURN '';
    END IF;
    RETURN E'\n\n📊 Yeni tahminler' || summary;
END;
$$;

GRANT EXECUTE ON FUNCTION public.format_diversified_prediction_markets(JSONB) TO service_role;

COMMIT;
