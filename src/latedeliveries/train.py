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
        pipeline = build_pipeline(model)
        scores = cross_validate(
            pipeline,
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

        # The cheapest total cost this model reaches on out-of-fold training
        # probabilities: the number the business actually pays.
        probabilities = cross_val_predict(
            pipeline,
            X_train,
            y_train,
            cv=CV,
            method="predict_proba",
            n_jobs=-1,
        )[:, 1]
        threshold, results = choose_threshold(y_train, probabilities)
        row["best_threshold"] = threshold
        row["lowest_cost"] = min(result["cost"] for result in results)

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

    # The two logistic regressions reach the lowest out-of-fold cost of all
    # candidates (within about 1% of each other) and beat the random forest on
    # both cost and ROC-AUC. They rank orders almost identically, so the
    # logistic family is chosen and class_weight is left to the grid search.
    untuned = comparison.set_index("model").loc["Logistic Regression"]
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

    # ROC-AUC is threshold-free: it measures how well late orders are ranked
    # above on-time ones. The threshold is then chosen from costs separately.
    gain = float(grid.best_score_ - untuned["roc_auc_mean"])
    tuning = {
        "scoring": "roc_auc",
        "untuned_cv_roc_auc_mean": float(untuned["roc_auc_mean"]),
        "untuned_cv_roc_auc_std": float(untuned["roc_auc_std"]),
        "tuned_cv_roc_auc_mean": float(grid.best_score_),
        "tuned_cv_roc_auc_std": float(
            grid.cv_results_["std_test_score"][grid.best_index_]
        ),
        "gain": gain,
        "gain_exceeds_fold_spread": bool(gain > untuned["roc_auc_std"]),
    }
    print("Best settings:", grid.best_params_)
    print(json.dumps(tuning, indent=2))

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
        "tuning": tuning,
        "test_results": metrics,
    }

    with (models / "model_info.json").open("w") as file:
        json.dump(model_info, file, indent=2)

    print("\nTest results:")
    print(json.dumps(metrics, indent=2))
    print("\nSaved model and reports successfully.")


if __name__ == "__main__":
    main()
