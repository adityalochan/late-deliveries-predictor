"""Shared fixtures."""

import pytest
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, split_data


@pytest.fixture(scope="session")
def split():
    """The stratified train/test split, built once per test session."""
    return split_data()


@pytest.fixture(scope="session")
def train_data(split):
    """Training features and labels only. Tests never use the test set."""
    X_train, _, y_train, _ = split
    return X_train, y_train


@pytest.fixture(scope="session")
def fitted_model(train_data):
    """A logistic regression pipeline fitted on the training data."""
    X_train, y_train = train_data
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)
    return model


@pytest.fixture
def small_example():
    """Five orders whose threshold costs are worked out by hand."""
    return [1, 1, 0, 0, 0], [0.9, 0.6, 0.4, 0.2, 0.1]
