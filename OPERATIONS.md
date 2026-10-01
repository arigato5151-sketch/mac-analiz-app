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

## Missing production model manifest

If `Daily data update` fails at `Generate upcoming predictions` with
`Artifact manifest is missing`, check the `latest.joblib` and
`artifact_manifest.json` objects in the private Supabase Storage `models` bucket
under `model_artifacts/`. Both must come from the same trusted training or
promotion run. Do not generate a new manifest from an existing unverified binary:
that would make the checksum check meaningless.

Recover the original manifest from the trusted artifact publication, or train and
validate a new model in an isolated run and publish its model and manifest
together. Only after the bundle verifies should the morning update be rerun.
The public Streamlit application cannot perform this recovery with its read-only
credentials. Nightly `Evaluate completed predictions` failures mentioning
`shadow_prediction_performance.id` come from an older checkout; the current
code orders that table by its actual primary key, `shadow_prediction_id`.
