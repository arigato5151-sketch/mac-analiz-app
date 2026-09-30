import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from models.train_model import refit_production_model


def test_production_refit_freezes_best_iteration_and_uses_all_rows():
    features = pd.DataFrame(np.arange(90, dtype=float).reshape(30, 3))
    labels = np.array([0, 1, 2] * 10)
    selected = XGBClassifier(
        objective="multi:softprob",
        num_class=3,
        n_estimators=8,
        max_depth=2,
        learning_rate=0.1,
        early_stopping_rounds=2,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=1,
        tree_method="hist",
    )
    selected.fit(features.iloc[:20], labels[:20], eval_set=[(features.iloc[20:], labels[20:])], verbose=False)

    production = refit_production_model(
        selected,
        features=features,
        labels=labels,
        weights=np.ones(len(labels)),
    )

    assert production.get_params()["n_estimators"] == selected.best_iteration + 1
    assert production.get_params()["early_stopping_rounds"] is None
