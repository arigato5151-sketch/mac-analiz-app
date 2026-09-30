# Operations and scheduling

GitHub Actions and `scheduler.py` are both supported. GitHub Actions is the source
of truth for scheduled production runs; `scheduler.py` remains available for local,
manual, and recovery runs.

Both paths must remain idempotent:

- Pre-match delivery uses the database claim/idempotency record.
- Coupon and prediction writes use unique keys and may be retried.
- Weekly retraining is single-instance: GitHub Actions concurrency and scheduler
  `max_instances=1` must remain enabled.
- The single pre-match deadline is
  `config.settings.PRE_MATCH_DECISION_LEAD_MINUTES` (currently 20 minutes).
  A delayed run compensates within the idempotent pre-match window.

The workflow owns production timing. The in-process scheduler is for manual/local
execution and recovery; it must not be run concurrently with production unless the
database idempotency/claim protections are available.
