-- Change the database dispatcher to the 20-minute pre-kickoff notification window.
BEGIN;

DO $$
DECLARE
    current_definition TEXT;
BEGIN
    SELECT pg_get_functiondef('public.dispatch_due_telegram_pre_match_alerts()'::REGPROCEDURE)
    INTO current_definition;

    IF current_definition IS NULL THEN
        RAISE EXCEPTION 'Telegram dispatcher must exist before its window can be updated';
    END IF;

    current_definition := REPLACE(current_definition, 'INTERVAL ''10 minutes''', 'INTERVAL ''15 minutes''');
    current_definition := REPLACE(current_definition, 'INTERVAL ''90 minutes''', 'INTERVAL ''25 minutes''');
    EXECUTE current_definition;
END;
$$;

COMMIT;
