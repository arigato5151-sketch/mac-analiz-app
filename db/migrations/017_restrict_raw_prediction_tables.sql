-- Keep raw prediction evaluation tables private; the public UI uses the
-- evaluated_prediction_results view instead.
BEGIN;

REVOKE ALL ON TABLE public.team_form, public.prediction_snapshots,
    public.prediction_performance
FROM anon, authenticated;

DROP POLICY IF EXISTS "public_read_team_form" ON public.team_form;
DROP POLICY IF EXISTS "public_read_prediction_snapshots"
    ON public.prediction_snapshots;
DROP POLICY IF EXISTS "public_read_prediction_performance"
    ON public.prediction_performance;

GRANT SELECT ON public.evaluated_prediction_results TO anon;

COMMIT;
