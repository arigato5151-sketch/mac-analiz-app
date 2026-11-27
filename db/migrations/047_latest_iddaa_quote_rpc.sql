-- Return only the newest IdDaa quote per fixture for the public UI and health check.
BEGIN;

CREATE OR REPLACE FUNCTION public.latest_iddaa_quotes(p_match_ids INTEGER[])
RETURNS TABLE (
    id BIGINT,
    match_id INTEGER,
    bookmaker TEXT,
    odds JSONB,
    captured_at TIMESTAMPTZ,
    source_updated_at TIMESTAMPTZ
)
LANGUAGE SQL
STABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $function$
    SELECT DISTINCT ON (quote.match_id)
        quote.id,
        quote.match_id,
        quote.bookmaker,
        quote.odds,
        quote.captured_at,
        quote.source_updated_at
    FROM public.odds_quote_history AS quote
    WHERE quote.bookmaker = 'iddaa.com'
      AND quote.match_id = ANY(COALESCE(p_match_ids, ARRAY[]::INTEGER[]))
    ORDER BY quote.match_id, quote.captured_at DESC, quote.id DESC;
$function$;

CREATE OR REPLACE FUNCTION public.latest_iddaa_quote_times(p_match_ids INTEGER[])
RETURNS TABLE (match_id INTEGER, captured_at TIMESTAMPTZ)
LANGUAGE SQL
STABLE
SECURITY INVOKER
SET search_path = pg_catalog, public
AS $function$
    SELECT DISTINCT ON (quote.match_id)
        quote.match_id,
        quote.captured_at
    FROM public.odds_quote_history AS quote
    WHERE quote.bookmaker = 'iddaa.com'
      AND quote.match_id = ANY(COALESCE(p_match_ids, ARRAY[]::INTEGER[]))
    ORDER BY quote.match_id, quote.captured_at DESC, quote.id DESC;
$function$;

REVOKE ALL ON FUNCTION public.latest_iddaa_quotes(INTEGER[]) FROM PUBLIC;
REVOKE ALL ON FUNCTION public.latest_iddaa_quote_times(INTEGER[]) FROM PUBLIC;
GRANT EXECUTE ON FUNCTION public.latest_iddaa_quotes(INTEGER[]) TO anon, service_role;
GRANT EXECUTE ON FUNCTION public.latest_iddaa_quote_times(INTEGER[]) TO anon, service_role;

COMMIT;
