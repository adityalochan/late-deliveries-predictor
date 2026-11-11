import numpy as np
import pytest

from latedeliveries.data import make_orders
from latedeliveries.pipeline import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    split_data,
)


@pytest.mark.parametrize(
    "excluded",
    ["order_id", "prep_minutes", "customer_rating", "late"],
)
def test_excluded_features(excluded):
    assert excluded not in FEATURES


def test_feature_lists_do_not_overlap():
    assert not set(NUMERIC_FEATURES) & set(CATEGORICAL_FEATURES)


def test_split_sizes_and_stratification(split):
    X_train, X_test, y_train, y_test = split
    assert len(X_train) == 2250
    assert len(X_test) == 750
    assert abs(y_train.mean() - y_test.mean()) < 0.01


def test_split_is_repeatable(split):
    again = split_data()
    for a, b in zip(split, again, strict=True):
        assert a.equals(b)


def test_probabilities_are_valid(fitted_model, train_data):
    X_train, _ = train_data
    probabilities = fitted_model.predict_proba(X_train)
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    np.testing.assert_allclose(probabilities.sum(axis=1), 1.0)


def test_missing_value_and_unseen_category(fitted_model, train_data):
    X_train, _ = train_data
    example = X_train.iloc[[0]].copy()
    example["weather"] = "hail"
    example["distance_km"] = np.nan
    probabilities = fitted_model.predict_proba(example)[:, 1]
    assert len(probabilities) == 1
    assert np.isfinite(probabilities).all()


def test_raw_order_with_missing_prep_minutes_and_unseen_weather(fitted_model):
    # A full order row as another program would send it: prep_minutes is
    # missing and the weather is a value never seen in training.
    order = make_orders(n=1).drop(columns="late")
    order["prep_minutes"] = np.nan
    order["weather"] = "fog"
    probability = fitted_model.predict_proba(order)[:, 1]
    assert probability.shape == (1,)
    assert 0 <= probability[0] <= 1


def test_data_generator():
    assert make_orders().shape[0] == 3000
    assert RANDOM_STATE == 42
