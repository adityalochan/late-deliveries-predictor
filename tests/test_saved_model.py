import joblib
import numpy as np


def test_saved_model_predictions_match(fitted_model, train_data, tmp_path):
    X_train, _ = train_data

    path = tmp_path / "test_model.joblib"
    joblib.dump(fitted_model, path)
    restored = joblib.load(path)

    before = fitted_model.predict_proba(X_train.iloc[:50])
    after = restored.predict_proba(X_train.iloc[:50])

    np.testing.assert_allclose(before, after)
