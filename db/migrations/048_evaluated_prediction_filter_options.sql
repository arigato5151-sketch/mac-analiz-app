-- Keep performance-screen filter metadata compact and apply selected filters in PostgREST.
BEGIN;

CREATE OR REPLACE FUNCTION public.evaluated_prediction_filter_options()
RETURNS TABLE (
    min_match_date TIMESTAMPTZ,
    max_match_date TIMESTAMPTZ,
    model_versions TEXT[],
    league_options JSONB
)
LANGUAGE SQL
STABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $function$
    SELECT
        MIN(result.match_date),
        MAX(result.match_date),
        COALESCE(
            ARRAY_AGG(DISTINCT result.model_version)
                FILTER (WHERE result.model_version IS NOT NULL),
            ARRAY[]::TEXT[]
        ),
        COALESCE(
            (
                SELECT JSONB_AGG(
                    JSONB_BUILD_OBJECT('id', league.league_id, 'name', league.league_name)
                    ORDER BY league.league_name, league.league_id
                )
                FROM (
                    SELECT DISTINCT league_id, league_name
                    FROM public.evaluated_prediction_results
                    WHERE league_id IS NOT NULL AND league_name IS NOT NULL
                ) AS league
            ),
            '[]'::JSONB
        )
    FROM public.evaluated_prediction_results AS result;
$function$;

REVOKE ALL ON FUNCTION public.evaluated_prediction_filter_options() FROM PUBLIC;
REVOKE ALL ON FUNCTION public.evaluated_prediction_filter_options() FROM authenticated;
GRANT EXECUTE ON FUNCTION public.evaluated_prediction_filter_options() TO anon, service_role;

COMMIT;
