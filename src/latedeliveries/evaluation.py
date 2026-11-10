
"""Evaluate the business cost of delivery predictions."""

import numpy as np

VOUCHER_COST = 4
MISSED_LATE_COST = 15


def order_cost(y_true, flagged):
    """Calculate the total cost of delivery decisions."""
    y_true = np.asarray(y_true)
    flagged = np.asarray(flagged, dtype=bool)

    if len(y_true) != len(flagged):
        raise ValueError("Actual outcomes and predictions must have equal lengths.")

    vouchers = flagged.sum()
    missed_late = ((y_true == 1) & ~flagged).sum()

    return int(vouchers * VOUCHER_COST + missed_late * MISSED_LATE_COST)
