# SQL migrations

Each numbered file is immutable once applied. `python -m db.migrations --validate` checks ordering, transaction boundaries, duplicate versions, and unexpected numeric gaps. Version `030` is intentionally reserved and is the only allowed gap; migrations are never renumbered. `--verify-production` also compares every checksum with Supabase's `schema_migrations` ledger.

Apply a new migration in this order:

1. Add one new numbered `.sql` file; never edit an applied file.
2. Run `python -m db.migrations --validate`.
3. Apply that exact file through the Supabase SQL editor.
4. Record the displayed SHA-256 checksum in `schema_migrations` in the same transaction.
5. Run `python -m db.migrations --verify-production`.

To refresh the repository migration manifest, run `python -m db.migrations --manifest`.

The CI workflow enforces repository validation. Production verification runs in the data workflow after credentials are available.
