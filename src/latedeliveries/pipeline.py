from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from latedeliveries.data import make_orders

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "distance_km",
    "items",
    "hour",
    "courier_months",
]

CATEGORICAL_FEATURES = [
    "weather",
    "day_type",
    "vehicle",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def split_data():
    """Create a repeatable, stratified 75/25 split."""
    data = make_orders()
    X = data[FEATURES]
    y = data["late"]

    return train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )


def build_pipeline(
    model,
    numeric_features=NUMERIC_FEATURES,
    categorical_features=CATEGORICAL_FEATURES,
):
    """Build a leak-free preprocessing and modelling pipeline.

    The feature lists default to the approved ones. They are parameters only
    so the notebook's leakage check can add an excluded column back on purpose.
    """
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore"),
            )
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_features)),
            ("categorical", categorical_pipeline, list(categorical_features)),
        ]
    )

    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            ("model", model),
        ]
    )
