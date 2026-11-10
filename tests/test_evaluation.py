
import numpy as np

from latedeliveries.evaluation import choose_threshold, order_cost


def test_cost_hand_calculation():
    y_true = [1, 0, 1, 0]
    flagged = [True, True, False, False]
    assert order_cost(y_true, flagged) == 23


def test_flag_nobody_is_cheapest():
    y_true = [0, 0, 0, 0]
    probabilities = [0.1, 0.2, 0.3, 0.4]
    threshold, _ = choose_threshold(y_true, probabilities)
    assert order_cost(
        y_true, np.asarray(probabilities) >= threshold
    ) == 0


def test_flag_everybody_is_cheapest():
    y_true = [1, 1, 1, 1]
    probabilities = [0.2, 0.4, 0.6, 0.8]
    threshold, _ = choose_threshold(y_true, probabilities)
    assert order_cost(
        y_true, np.asarray(probabilities) >= threshold
    ) == 16


def test_threshold_is_from_required_grid():
    threshold, results = choose_threshold(
        [1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2]
    )
    assert 0.05 <= threshold <= 0.95
    assert len(results) == 19
