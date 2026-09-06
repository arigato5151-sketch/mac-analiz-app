-- Allow public dashboards to query the narrow performance view while keeping
-- the underlying prediction_performance table private.
BEGIN;

ALTER VIEW public.live_prediction_performance
SET (security_invoker = false);

GRANT SELECT ON public.live_prediction_performance,
    public.evaluated_prediction_results
TO anon, service_role;

COMMIT;
