-- Durable result-message delivery queue. Rows are retained until Telegram
-- confirms delivery, so transient failures are retried on later night runs.
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

COMMIT;
