-- Make availability counts explicit: available_count is available players,
-- unavailable_count is injured, suspended, or doubtful players.
BEGIN;

ALTER TABLE public.team_availability_status
    ADD COLUMN IF NOT EXISTS unavailable_count INTEGER;

UPDATE public.team_availability_status
SET unavailable_count = available_count
WHERE unavailable_count IS NULL;

UPDATE public.team_availability_status
SET available_count = GREATEST(0, 22 - unavailable_count)
WHERE unavailable_count IS NOT NULL;

ALTER TABLE public.team_availability_status
    ALTER COLUMN available_count SET DEFAULT 22,
    ALTER COLUMN unavailable_count SET DEFAULT 0,
    ALTER COLUMN unavailable_count SET NOT NULL;

ALTER TABLE public.team_availability_history
    ADD COLUMN IF NOT EXISTS unavailable_count INTEGER;

UPDATE public.team_availability_history
SET unavailable_count = available_count
WHERE unavailable_count IS NULL;

UPDATE public.team_availability_history
SET available_count = GREATEST(0, 22 - unavailable_count)
WHERE unavailable_count IS NOT NULL;

ALTER TABLE public.team_availability_history
    ALTER COLUMN unavailable_count SET DEFAULT 0,
    ALTER COLUMN unavailable_count SET NOT NULL;

COMMIT;
