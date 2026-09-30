from models.feature_drift_report import build_report


def test_report_marks_sparse_features_without_removing_them():
    report = build_report(
        [{"home_available_count": 20, "home_lineup_confirmed": True}],
        [{"home_available_count": None, "home_lineup_confirmed": False}],
        minimum_coverage=0.8,
    )

    assert report["features"]["home_available_count"]["training"]["coverage"] == 1.0
    assert report["features"]["home_available_count"]["live"]["low_coverage"] is True
    assert any("home_available_count" in item for item in report["recommendations"])


def test_report_handles_empty_snapshots():
    report = build_report([], [])
    assert report["training_rows"] == 0
    assert report["features"]["away_impact_score"]["live"]["coverage"] == 0.0
