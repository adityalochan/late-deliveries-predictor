import numpy as np
import pytest

from latedeliveries.evaluation import choose_threshold, order_cost


def test_cost_hand_calculation():
    y_true = [1, 0, 1, 0]
    flagged = [True, True, False, False]
    assert order_cost(y_true, flagged) == 23


def test_flag_nobody_is_cheapest():
    y_true = [0, 0, 0, 0]
    probabilities = [0.1, 0.2, 0.3, 0.4]
    threshold, _ = choose_threshold(y_true, probabilities)
    assert order_cost(y_true, np.asarray(probabilities) >= threshold) == 0


def test_flag_everybody_is_cheapest():
    y_true = [1, 1, 1, 1]
    probabilities = [0.2, 0.4, 0.6, 0.8]
    threshold, _ = choose_threshold(y_true, probabilities)
    assert order_cost(y_true, np.asarray(probabilities) >= threshold) == 16


def test_threshold_is_from_required_grid():
    threshold, results = choose_threshold([1, 0, 1, 0], [0.9, 0.1, 0.8, 0.2])
    assert 0.05 <= threshold <= 0.95
    assert len(results) == 19


def test_hand_worked_middle_threshold(small_example):
    # Worked on paper for labels [1, 1, 0, 0, 0]:
    #   t <= 0.10       flag 5, miss 0 -> 5 x 4          = 20
    #   t = 0.25..0.40  flag 3, miss 0 -> 3 x 4          = 12
    #   t = 0.45..0.60  flag 2, miss 0 -> 2 x 4          =  8  cheapest
    #   t = 0.65..0.90  flag 1, miss 1 -> 1 x 4 + 1 x 15 = 19
    #   t = 0.95        flag 0, miss 2 -> 2 x 15         = 30
    y_true, probabilities = small_example
    threshold, results = choose_threshold(y_true, probabilities)
    costs = {row["threshold"]: row["cost"] for row in results}
    assert threshold == 0.45
    assert costs[0.05] == 20
    assert costs[0.3] == 12
    assert costs[0.7] == 19
    assert costs[0.95] == 30


def test_cost_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="same shape"):
        order_cost([1, 0, 1], [True, False])
