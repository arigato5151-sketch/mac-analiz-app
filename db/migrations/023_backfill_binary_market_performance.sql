-- Backfill binary-market evaluations that predate migration 002.
BEGIN;

WITH evaluation_source AS (
    SELECT
        performance.prediction_id,
        COALESCE(snapshot.prob_over_2_5, prediction.prob_over_2_5) AS prob_over_2_5,
        COALESCE(snapshot.prob_btts, prediction.prob_btts) AS prob_btts,
        (fixture.home_score + fixture.away_score >= 3) AS over_2_5_actual,
        (fixture.home_score > 0 AND fixture.away_score > 0) AS btts_actual
    FROM public.prediction_performance AS performance
    JOIN public.predictions AS prediction
        ON prediction.id = performance.prediction_id
    JOIN public.matches AS fixture
        ON fixture.id = prediction.match_id
    LEFT JOIN public.prediction_snapshots AS snapshot
        ON snapshot.id = performance.snapshot_id
    WHERE fixture.home_score IS NOT NULL
      AND fixture.away_score IS NOT NULL
)
UPDATE public.prediction_performance AS performance
SET
    over_2_5_actual = CASE
        WHEN source.prob_over_2_5 BETWEEN 0 AND 1
        THEN COALESCE(performance.over_2_5_actual, source.over_2_5_actual)
        ELSE performance.over_2_5_actual
    END,
    over_2_5_was_correct = CASE
        WHEN source.prob_over_2_5 BETWEEN 0 AND 1
        THEN COALESCE(
            performance.over_2_5_was_correct,
            (source.prob_over_2_5 >= 0.5) = source.over_2_5_actual
        )
        ELSE performance.over_2_5_was_correct
    END,
    over_2_5_brier_score = CASE
        WHEN source.prob_over_2_5 BETWEEN 0 AND 1
        THEN COALESCE(
            performance.over_2_5_brier_score,
            POWER(
                source.prob_over_2_5 - source.over_2_5_actual::INTEGER,
                2
            )
        )
        ELSE performance.over_2_5_brier_score
    END,
    btts_actual = CASE
        WHEN source.prob_btts BETWEEN 0 AND 1
        THEN COALESCE(performance.btts_actual, source.btts_actual)
        ELSE performance.btts_actual
    END,
    btts_was_correct = CASE
        WHEN source.prob_btts BETWEEN 0 AND 1
        THEN COALESCE(
            performance.btts_was_correct,
            (source.prob_btts >= 0.5) = source.btts_actual
        )
        ELSE performance.btts_was_correct
    END,
    btts_brier_score = CASE
        WHEN source.prob_btts BETWEEN 0 AND 1
        THEN COALESCE(
            performance.btts_brier_score,
            POWER(source.prob_btts - source.btts_actual::INTEGER, 2)
        )
        ELSE performance.btts_brier_score
    END
FROM evaluation_source AS source
WHERE performance.prediction_id = source.prediction_id
  AND (
      (
          source.prob_over_2_5 BETWEEN 0 AND 1
          AND (
              performance.over_2_5_actual IS NULL
              OR performance.over_2_5_was_correct IS NULL
              OR performance.over_2_5_brier_score IS NULL
          )
      )
      OR (
          source.prob_btts BETWEEN 0 AND 1
          AND (
              performance.btts_actual IS NULL
              OR performance.btts_was_correct IS NULL
              OR performance.btts_brier_score IS NULL
          )
      )
  );

COMMIT;
