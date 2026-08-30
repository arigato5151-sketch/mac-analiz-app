-- Migration ledger bootstrap. Apply once before recording any migrations.
BEGIN;

CREATE TABLE IF NOT EXISTS public.schema_migrations (
    version TEXT PRIMARY KEY,
    checksum TEXT NOT NULL,
    applied_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

GRANT SELECT, INSERT, UPDATE ON public.schema_migrations TO service_role;

COMMIT;
