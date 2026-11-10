
import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression

from latedeliveries.pipeline import build_pipeline, split_data


def test_saved_model_predictions_match(tmp_path):
    X_train, _, y_train, _ = split_data()
    model = build_pipeline(LogisticRegression(max_iter=2000))
    model.fit(X_train, y_train)

    path = tmp_path / "test_model.joblib"
    joblib.dump(model, path)

    restored = joblib.load(path)

    before = model.predict_proba(X_train.iloc[:10])
    after = restored.predict_proba(X_train.iloc[:10])

    np.testing.assert_allclose(before, after)
