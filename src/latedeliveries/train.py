"""Train and evaluate the late-delivery prediction model."""

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import (
    GridSearchCV,
    StratifiedKFold,
    cross_val_predict,
    cross_validate,
)

from latedeliveries.evaluation import choose_threshold, evaluate_predictions
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_data,
)

CV = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=RANDOM_STATE,
)

SCORING = {
    "accuracy": "accuracy",
    "precision": "precision",
    "recall": "recall",
    "f1": "f1",
    "roc_auc": "roc_auc",
}


def compare_models(X_train, y_train):
    """Compare baseline and candidate models using training-only CV."""
    candidates = {
        "Dummy": DummyClassifier(strategy="prior"),
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Balanced Logistic Regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            random_state=RANDOM_STATE,
        ),
    }

    rows = []

    for name, model in candidates.items():
        scores = cross_validate(
            build_pipeline(model),
            X_train,
            y_train,
            cv=CV,
            scoring=SCORING,
            n_jobs=-1,
        )

        row = {"model": name}

        for metric in SCORING:
            values = scores[f"test_{metric}"]
            row[f"{metric}_mean"] = float(np.mean(values))
            row[f"{metric}_std"] = float(np.std(values))

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    """Run the complete reproducible training workflow."""
    root = Path(__file__).resolve().parents[2]
    reports = root / "reports"
    models = root / "models"

    reports.mkdir(exist_ok=True)
    models.mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = split_data()

    print("Comparing models...")
    comparison = compare_models(X_train, y_train)
    comparison.to_csv(reports / "cv_results.csv", index=False)
    print(comparison.to_string(index=False))

    # Logistic regression is interpretable, efficient, and produces
    # probabilities suitable for the voucher-cost decision rule.
    selected = build_pipeline(
        LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        )
    )

    print("\nTuning logistic regression...")
    grid = GridSearchCV(
        selected,
        param_grid={
            "model__C": [0.1, 1.0, 10.0],
            "model__class_weight": [None, "balanced"],
        },
        scoring="roc_auc",
        cv=CV,
        n_jobs=-1,
    )
    grid.fit(X_train, y_train)

    print("Best settings:", grid.best_params_)
    print("Best CV ROC-AUC:", grid.best_score_)

    # Threshold selection uses out-of-fold training probabilities.
    # The test set is not involved.
    print("\nSelecting cost-based threshold...")
    probabilities_cv = cross_val_predict(
        grid.best_estimator_,
        X_train,
        y_train,
        cv=CV,
        method="predict_proba",
        n_jobs=-1,
    )[:, 1]

    threshold, threshold_results = choose_threshold(y_train, probabilities_cv)
    print("Chosen threshold:", threshold)

    pd.DataFrame(threshold_results).to_csv(reports / "threshold_costs.csv", index=False)

    # Only now evaluate on the held-out test set.
    final_pipeline = grid.best_estimator_
    final_pipeline.fit(X_train, y_train)
    test_probabilities = final_pipeline.predict_proba(X_test)[:, 1]

    metrics = evaluate_predictions(y_test, test_probabilities, threshold)

    with (reports / "metrics.json").open("w") as file:
        json.dump(metrics, file, indent=2)

    joblib.dump(final_pipeline, models / "late_orders.joblib")

    model_info = {
        "sklearn_version": sklearn.__version__,
        "seed": RANDOM_STATE,
        "threshold": threshold,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "best_settings": grid.best_params_,
        "cv_roc_auc": float(grid.best_score_),
        "test_results": metrics,
    }

    with (models / "model_info.json").open("w") as file:
        json.dump(model_info, file, indent=2)

    print("\nTest results:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model and reports successfully.")


if __name__ == "__main__":
    main()
