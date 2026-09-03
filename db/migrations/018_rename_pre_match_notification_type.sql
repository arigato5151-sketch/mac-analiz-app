-- Align the notification log label with the actual 15-25 minute window.
BEGIN;

UPDATE public.notification_log
SET notification_type = 'pre_match_20m'
WHERE notification_type = 'pre_match_60m';

COMMIT;
