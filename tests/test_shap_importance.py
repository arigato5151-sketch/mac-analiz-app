import numpy as np
import pandas as pd

from models import train_model


def test_shap_importance_supports_three_dimensional_multiclass_output(monkeypatch):
    class FakeExplainer:
        def __init__(self, _model):
            pass

        def shap_values(self, features):
            return np.ones((len(features), 2, 3))

    monkeypatch.setattr(train_model, "SHAP_AVAILABLE", True)
    monkeypatch.setattr(train_model.shap, "TreeExplainer", FakeExplainer)

    rows, status = train_model._compute_shap_importance(
        object(), pd.DataFrame([[1, 2], [3, 4]]), ["a", "b"]
    )

    assert status == "ok"
    assert [row["feature"] for row in rows] == ["a", "b"]
