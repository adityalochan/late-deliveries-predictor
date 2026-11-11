# Late Deliveries

QuickBite wants to know, at the moment an order is placed, whether it will
arrive late, so it can send a $4 voucher before the customer is disappointed.
This project trains a scikit-learn pipeline that predicts the probability that
an order will be late, and flags it when that probability passes a threshold
chosen from the company's own costs.

## Setup

```bash
conda env create -f environment.yml
conda activate lab05-late-deliveries
pip install -r requirements.txt
pip install -e .
```

## Run

```bash
python -m latedeliveries.train
```

This recreates `reports/cv_results.csv`, `reports/threshold_costs.csv`,
`reports/metrics.json`, `models/late_orders.joblib` and `models/model_info.json`.
The analysis is in `notebooks/analysis.ipynb`, saved with its outputs.

## Check it

```bash
python -m ruff format --check .
python -m ruff check .
python -m pytest
```

## Decisions

### Columns

The model is used when the order is placed, so a column is only allowed if its
value is known at that moment.

| Used | Type |
|---|---|
| `distance_km`, `items`, `hour`, `courier_months` | numeric (median-imputed, scaled) |
| `weather`, `day_type`, `vehicle` | categorical (one-hot, unknown values ignored) |

| Left out | Reason |
|---|---|
| `order_id` | An arbitrary label with no meaning for lateness; any pattern in it would be coincidence. |
| `prep_minutes` | The time the kitchen *actually* took, which is only known after the food is cooked, not when the order is placed. |
| `customer_rating` | Given after delivery, and driven by whether the order was late. Using it leaks the answer. |
| `late` | The target. |

**Leakage check** (training data only, 5-fold CV, logistic regression):

| Features | ROC-AUC mean | std | change |
|---|---|---|---|
| approved features | 0.779 | 0.038 | — |
| + `prep_minutes` | 0.795 | 0.034 | +0.016 |
| + `customer_rating` | 0.905 | 0.021 | **+0.125** |

Adding `customer_rating` back lifts ROC-AUC sharply, from 0.78 to 0.91. That is
a warning, not good news: the column would not exist at prediction time. Both
columns were taken out again.

### Model

Five-fold stratified cross-validation on the 2,250 training orders only.
`lowest_cost` is the cheapest total cost each model reaches with its
out-of-fold probabilities across the threshold grid ($4 per flag + $15 per
missed late order).

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | Lowest cost |
|---|---|---|---|---|---|---|
| Dummy (prior) | 0.844 ± 0.001 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.000 ± 0.000 | 0.500 ± 0.000 | $5,265 |
| Logistic Regression | 0.862 ± 0.013 | 0.669 ± 0.114 | 0.222 ± 0.066 | 0.330 ± 0.086 | 0.779 ± 0.038 | $4,205 |
| Balanced Logistic Regression | 0.731 ± 0.026 | 0.323 ± 0.035 | 0.655 ± 0.054 | 0.432 ± 0.040 | 0.779 ± 0.038 | $4,183 |
| Random Forest | 0.853 ± 0.009 | 0.568 ± 0.069 | 0.216 ± 0.066 | 0.310 ± 0.078 | 0.736 ± 0.030 | $4,494 |

The Dummy is 84% accurate by never flagging anyone, which shows why accuracy
cannot choose the model. I chose **logistic regression** because of cost, not
the highest single metric:

* Both logistic regressions reach the lowest cost, about $1,070 cheaper than the
  baseline and about $300 cheaper than the random forest.
* They differ by only $22 (0.5%) and rank orders almost identically
  (ROC-AUC 0.779 for both). `class_weight` mainly moves where the best threshold
  sits, so it was left to the grid search to decide.
* Logistic regression is also simple to explain, and it gives probabilities
  that the threshold can act on directly.

### Tuning

`GridSearchCV` over `model__C` ∈ {0.1, 1, 10} and `model__class_weight` ∈
{None, "balanced"} (6 combinations), with the same 5 folds.

* **Scoring: ROC-AUC.** It is threshold-free and measures how well late orders
  are ranked above on-time ones. The threshold is then chosen separately from
  costs, so ranking quality is what tuning should improve.
* **Best settings:** `C = 0.1`, `class_weight = None`.
* **Did tuning help?** No. The CV ROC-AUC went from 0.7795 to 0.7798, a gain of
  0.0003. That is far smaller than the fold-to-fold spread of 0.038, so tuning
  did not beat the untuned model by more than the noise.

### Threshold

The default 0.5 treats both mistakes as equally bad, but here missing a late
order ($15) costs almost four times as much as an unneeded voucher ($4). The
cost rule needs its own threshold.

I took out-of-fold probabilities on the training data with `cross_val_predict`.
I then computed the total cost at every threshold from 0.05 to 0.95 in steps of
0.05 (`reports/threshold_costs.csv`) and kept the cheapest: **0.30**, with a
cost of $4,175. 0.50 would have cost $4,605. 0.30 is close to the break-even
probability 4 / 15 ≈ 0.27, where a voucher costs the same as the expected loss
of not sending one. The test set played no part in this choice.

## Result

The final pipeline was fitted on the whole training set and scored **once** on
the 750 test orders, at threshold 0.30.

|  | Not flagged | Flagged |
|---|---|---|
| **On time** | 589 | 44 |
| **Late** | 72 | 45 |

| Precision | Recall | F1 | ROC-AUC | Accuracy |
|---|---|---|---|---|
| 0.506 | 0.385 | 0.437 | 0.781 | 0.845 |

| Policy | Total cost |
|---|---|
| Flag nobody (today) | $1,755 |
| Flag everybody | $3,000 |
| **Use the model** | **$1,436** |

The model saves $319 (18%) against today's policy. The test ROC-AUC (0.781)
matches the CV estimate (0.780), so the result is not a lucky split. The size of
the saving still rests on only 117 late test orders, so I would trust the
direction but not the exact figure: another sample could move it by a few
hundred dollars.

## What the tests cover

18 tests (`python -m pytest`), sharing fixtures from `tests/conftest.py`:

* **Features:** the excluded columns never appear in the feature lists
  (parametrized), and the numeric and categorical lists don't overlap.
* **Split:** it is 75/25, stratified (same late rate in both parts), and
  repeatable.
* **Pipeline:** probabilities are valid. An order with a missing value and an
  unseen weather still gets a prediction, including a raw order with
  `prep_minutes` missing.
* **Cost and threshold:** both functions are checked on hand-worked examples. In
  one, flagging nobody is cheapest; in another, flagging everybody is. A
  five-order example has its cost at every threshold worked out on paper.
  Mismatched inputs raise an error.
* **Saved model:** a pipeline saved with joblib predicts the same after
  reloading (`tmp_path`).

Not tested: the exact numbers from the full training run. They depend on the
scikit-learn version, and pinning them in a test would make it fail on any
harmless library upgrade. Instead, `models/model_info.json` records the
version, and the committed reports can be compared with a fresh run.

## Known limitations

* The data is invented. The costs, the 0.30 threshold and the 18% saving would
  need checking against real orders before rollout.
* The model catches only 38% of late orders. Better features known at order
  time, such as live traffic, restaurant load or an *estimated* prep time,
  would likely help more than more tuning.
* Only logistic regression was tuned, over a small grid.
* The saving comes from a single test set of 750 orders.
