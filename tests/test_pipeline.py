
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.data import make_orders
from latedeliveries.pipeline import (
    FEATURES,
    RANDOM_STATE,
    build_pipeline,
    split_data,
)


@pytest.mark.parametrize(
    "excluded",
    ["order_id", "prep_minutes", "customer_rating", "late"],
)
def test_excluded_features(excluded):
    assert excluded not in FEATURES


def test_split_sizes_and_stratification():
    X_train, X_test, y_train, y_test = split_data()
    assert len(X_train) == 2250
    assert len(X_test) == 750
    assert abs(y_train.mean() - y_test.mean()) < 0.02


def test_split_is_repeatable():
    first = split_data()
    second = split_data()
    for a, b in zip(first, second):
        assert a.equals(b)


def test_probabilities_are_valid():
    X_train, _, y_train, _ = split_data()
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_train.iloc[:10])[:, 1]
    assert np.all((probabilities >= 0) & (probabilities <= 1))


def test_missing_value_and_unseen_category():
    X_train, _, y_train, _ = split_data()
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)

    example = X_train.iloc[[0]].copy()
    example["weather"] = "hail"
    example["distance_km"] = np.nan
    probabilities = model.predict_proba(example)[:, 1]
    assert len(probabilities) == 1
    assert np.isfinite(probabilities).all()


def test_data_generator():
    assert make_orders().shape[0] == 3000
    assert RANDOM_STATE == 42
