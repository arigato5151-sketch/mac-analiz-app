# SQL migrations

Each numbered file is immutable once applied. `python -m db.migrations --validate` checks ordering, transaction boundaries, duplicate versions, and unexpected numeric gaps. Version `030` is intentionally reserved and is the only allowed gap; migrations are never renumbered. `--verify-production` also compares every checksum with Supabase's `schema_migrations` ledger.

Apply a new migration in this order:

1. Add one new numbered `.sql` file; never edit an applied file.
2. Run `python -m db.migrations --validate`.
3. Run `python -m db.migrations --manifest` and copy the migration's displayed controller checksum. It is SHA-256 over `Path.read_text(encoding="utf-8")` re-encoded as UTF-8, so Python's universal-newline handling normalizes CRLF and LF to the same `\n` text. Do not use a hash of the raw file bytes.
4. In one Supabase SQL Editor transaction, execute the migration's statements between its top-level `BEGIN;` and `COMMIT;`, then insert its version and checksum into `public.schema_migrations`, and commit once. Do not execute the file's own `COMMIT;` before inserting the ledger row.
5. Run `python -m db.migrations --verify-production`.

For example, the transaction submitted to the SQL Editor has this shape:

```sql
BEGIN;
-- Paste the migration body here, excluding its outer BEGIN; and COMMIT;.
INSERT INTO public.schema_migrations (version, checksum)
VALUES ('041_revoke_dangerous_public_table_acl', '<checksum from --manifest>');
COMMIT;
```

To refresh the repository migration manifest, run `python -m db.migrations --manifest`.

The pinned checksums in `tests/fixtures/migration_checksums.json` protect migration source files from accidental edits in local CI. When adding a new migration, add its controller checksum to that fixture; never replace an applied migration's checksum. The production ledger comparison remains authoritative for deployed migrations: changing an applied file makes `--verify-production` report that version in `mismatched` and exit unsuccessfully.

## Nightly match analysis cache

Migration `043_nightly_match_analysis_cache.sql` creates a public-read, service-role-write cache of upcoming match forecasts, halftime/corner models, and each team's last five completed matches. The `Match statistics sync` workflow refreshes it once on the `30 20 * * *` UTC schedule (23:30 in Istanbul), after fixture and result updates. The Streamlit card reads this prepared JSON instead of loading full league history and running the model/backtest when opened. If a fixture was added or changed after the nightly refresh, the app falls back to the current on-demand calculation until the next scheduled refresh. The cache table retains only upcoming matches; the nightly job removes rows for past kickoff times before upserting the current cache.

## Production ACL verification

Migration `042_read_only_public_acl_check.sql` adds the read-only `public.verify_public_table_acl()` RPC. It returns one row per remaining effective `TRUNCATE`, `REFERENCES`, `TRIGGER`, or PostgreSQL 17+ `MAINTAIN` privilege for `anon` or `authenticated` on the nine tables governed by migration 041. Execution is revoked from `PUBLIC`, `anon`, and `authenticated`, and granted only to `service_role`. `python -m db.verify_acl` calls it; the workflow runs this step after verifying the migration ledger. The check sends the Supabase URL and service-role key only to that step's process environment and never prints them.

For a manual, seed-free SQL Editor check, both queries should return zero rows:

```sql
SELECT grantee, table_name, privilege_type
FROM information_schema.role_table_grants
WHERE table_schema = 'public'
  AND grantee IN ('anon', 'authenticated')
  AND table_name IN (
      'schema_migrations', 'result_notification_queue', 'notification_log',
      'model_candidates', 'shadow_predictions',
      'shadow_prediction_performance', 'personal_match_workspace',
      'personal_match_decisions', 'bookmaker_odds_snapshots'
  )
  AND privilege_type IN ('TRUNCATE', 'REFERENCES', 'TRIGGER')
ORDER BY grantee, table_name, privilege_type;
```

```sql
SELECT r.grantee, t.table_name, 'MAINTAIN' AS privilege_type
FROM (VALUES ('anon'), ('authenticated')) AS r(grantee)
CROSS JOIN (VALUES
    ('schema_migrations'), ('result_notification_queue'), ('notification_log'),
    ('model_candidates'), ('shadow_predictions'),
    ('shadow_prediction_performance'), ('personal_match_workspace'),
    ('personal_match_decisions'), ('bookmaker_odds_snapshots')
) AS t(table_name)
WHERE current_setting('server_version_num')::integer >= 170000
  AND has_table_privilege(r.grantee::name, format('public.%I', t.table_name), 'MAINTAIN')
ORDER BY r.grantee, t.table_name;
```

The CI workflow enforces repository validation. Its production steps only read the migration ledger and ACL state; they do not apply SQL or insert ledger rows. Production checks run after credentials are available and block later pipeline steps while the repository, ledger, and ACL state disagree.
