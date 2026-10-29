# SQL migrations

Each numbered file is immutable once applied. `python -m db.migrations --validate` checks ordering, transaction boundaries, duplicate versions, and unexpected numeric gaps. Version `030` is intentionally reserved and is the only allowed gap; migrations are never renumbered. `--verify-production` also compares every checksum with Supabase's `schema_migrations` ledger.

Apply a new migration in this order:

1. Add one new numbered `.sql` file; never edit an applied file.
2. Run `python -m db.migrations --validate`.
3. Run `python -m db.migrations --manifest` and copy the migration's displayed SHA-256 checksum.
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

The CI workflow enforces repository validation. Its production step only reads and compares the ledger; it does not apply SQL or insert ledger rows. Production verification runs in the data workflow after credentials are available and blocks later pipeline steps while the repository and ledger disagree.
