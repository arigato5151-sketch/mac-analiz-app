# SQL migrations

Each numbered file is immutable once applied. `python -m db.migrations --validate` checks ordering, transaction boundaries and duplicate versions. `--verify-production` also compares every checksum with Supabase's `schema_migrations` ledger.

Apply a new migration in this order:

1. Add one new numbered `.sql` file; never edit an applied file.
2. Run `python -m db.migrations --validate`.
3. Apply that exact file through the Supabase SQL editor.
4. Record the displayed SHA-256 checksum in `schema_migrations` in the same transaction.
5. Run `python -m db.migrations --verify-production`.

The CI workflow enforces repository validation. Production verification runs in the data workflow after credentials are available.
