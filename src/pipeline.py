
"""Feature selection, data splitting, and preprocessing pipeline."""

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

SEED = 42

NUMERIC_FEATURES = ["distance_km", "items", "hour"]
CATEGORICAL_FEATURES = ["weather", "day_type"]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data():
    """Create a repeatable, stratified train/test split."""
    orders = make_orders()

    X = orders[FEATURES]
    y = orders["late"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=SEED,
    )


def build_pipeline(model):
    """Combine preprocessing and a classification model."""
    preprocessing = ColumnTransformer(
        [
            (
                "numeric",
                make_pipeline(
                    SimpleImputer(strategy="median"),
                    StandardScaler(),
                ),
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    return Pipeline(
        [
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
