-- Track the origin of expected-goals metrics so secondary providers can fill
-- gaps without silently mixing or overwriting provider values.
BEGIN;

ALTER TABLE public.matches
    ADD COLUMN IF NOT EXISTS home_xg_source TEXT,
    ADD COLUMN IF NOT EXISTS away_xg_source TEXT,
    ADD COLUMN IF NOT EXISTS home_xa_source TEXT,
    ADD COLUMN IF NOT EXISTS away_xa_source TEXT;

UPDATE public.matches
SET home_xg_source = 'api-football'
WHERE home_xg IS NOT NULL AND home_xg_source IS NULL;

UPDATE public.matches
SET away_xg_source = 'api-football'
WHERE away_xg IS NOT NULL AND away_xg_source IS NULL;

UPDATE public.matches
SET home_xa_source = 'api-football'
WHERE home_xa IS NOT NULL AND home_xa_source IS NULL;

UPDATE public.matches
SET away_xa_source = 'api-football'
WHERE away_xa IS NOT NULL AND away_xa_source IS NULL;

COMMIT;
