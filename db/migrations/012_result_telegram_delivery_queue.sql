-- Queue evaluated fixtures before delivery so a failed Telegram request is retryable.
BEGIN;

CREATE TABLE IF NOT EXISTS public.result_notification_queue (
    prediction_id BIGINT PRIMARY KEY REFERENCES public.predictions(id) ON DELETE CASCADE,
    match_id INTEGER NOT NULL REFERENCES public.matches(id) ON DELETE CASCADE,
    available_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    sent_at TIMESTAMPTZ,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
    next_attempt_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    last_error TEXT
);

CREATE INDEX IF NOT EXISTS idx_result_notification_queue_pending
    ON public.result_notification_queue(next_attempt_at)
    WHERE sent_at IS NULL;

GRANT SELECT, INSERT, UPDATE, DELETE ON public.result_notification_queue TO service_role;
REVOKE ALL ON public.result_notification_queue FROM anon, authenticated;

CREATE OR REPLACE FUNCTION public.queue_result_telegram_notification()
RETURNS TRIGGER
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = public, pg_temp
AS $$
BEGIN
    -- The primary key makes repeated evaluation runs safe and prevents duplicates.
    INSERT INTO public.result_notification_queue (
        prediction_id, match_id, available_at, next_attempt_at
    )
    VALUES (NEW.prediction_id, NEW.match_id, NOW(), NOW())
    ON CONFLICT (prediction_id) DO NOTHING;

    RETURN NEW;
END;
$$;

CREATE TRIGGER queue_result_telegram_notification
AFTER INSERT ON public.prediction_performance
FOR EACH ROW EXECUTE FUNCTION public.queue_result_telegram_notification();

-- Recover finalised fixtures from the previous day that were evaluated before this trigger existed.
INSERT INTO public.result_notification_queue (
    prediction_id, match_id, available_at, next_attempt_at
)
SELECT performance.prediction_id, performance.match_id, performance.evaluated_at, NOW()
FROM public.prediction_performance AS performance
JOIN public.matches AS match_data ON match_data.id = performance.match_id
WHERE match_data.status = 'finished'
  AND match_data.match_date >= NOW() - INTERVAL '24 hours'
ON CONFLICT (prediction_id) DO NOTHING;

COMMIT;
