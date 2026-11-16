# Deviations from the protocol

Every place where the implementation differs from [METHOD.md](METHOD.md),
with its reason. All were decided before the holdout was opened.

## Dependencies (METHOD.md section 1)

| Deviation | Reason |
| --- | --- |
| `pyyaml` is declared. | Every command reads `configs/base.yaml`; no package of the planned stack parses YAML. It is a dependency of Optuna. |
| `scipy` is declared and imported directly (`models/har.py`, `evaluation.py`). | Sparse one-hot design matrix for the stock fixed effects; Student t distribution for the Diebold-Mariano p-value. It is a dependency of scikit-learn. |
| `matplotlib` is declared. | The report has five figures and the planned stack has no plotting library. |
| SHAP values come from LightGBM's own TreeSHAP (`predict(pred_contrib=True)`) when `shap` cannot be imported. | On the build machine `shap` fails to import: its `numba` dependency needs NumPy <= 2.3 and NumPy 2.4.6 is installed. TreeExplainer calls the same LightGBM routine, so the values are the same. `results/shap.csv` records which path was used. |

## Time-order recovery (METHOD.md 5.1)

| Deviation | Reason |
| --- | --- |
| Step 1: the tick is searched as the largest value on which at least 98% of a bucket's price levels are integer multiples (the smallest gap divided by 1 to 4), then refined by least squares. | The smallest gap alone carries float32 noise of 1-2% for high-priced stocks, and it can span several ticks. |
| Step 1: outliers are flagged twice. Before ordering, only a P0 more than 5 times away from the stock median. After a first ordering pass, any price more than 30% away from the rolling median of its 21 neighbours in that order; then the order is recovered again. | A P0 off by an integer factor cannot be told from a real price move before the order is known: stock prices range over a factor of about 2 across the sample. A per-stock rule flagged real prices. |
| Step 2: missing prices are not imputed with the column median. Distances ignore missing cells and the 20% largest squared differences of each pair. | A median-imputed price of a trending stock is a jump far larger than the distance between neighbouring time_ids. With imputation, 38 of 112 stocks were dropped and the two direction distances were equal to two digits (0.704 against 0.710). |
| Step 3: two methods were added to the two planned ones (Isomap on a k-NN graph; greedy path with 2-opt): `isomap_2opt` (Isomap, then 2-opt and single-row moves) and `end_start_chain` (a directed path that minimises the distance from the end of one time_id to the start of the next, built from the cheapest edges and refined by moving segments of 1 to 30 time_ids). `order.method: auto` selects by the synthetic test; it selected `end_start_chain`. | On the real data the start-price embedding linked only 54% of time_ids to their closest successor; the end-to-start distance is the cue of michaelpoluektov/orvp. All four methods are reported in `results/order_validation.md`. |
| Step 4: with `end_start_chain` the direction comes from the asymmetric cost itself. Both planned cues are still computed and reported, and they agree. | Consequence of the method. |
| Step 5: agreement is reported between four methods, not two, together with the end-to-start edge length, the best-successor share and the RV autocorrelation of each. | More methods. |
| The synthetic generator has days (13 buckets, then an overnight gap) and a volatility that changes between a feature window and its target window. | Without days, the planned methods pass the synthetic test and fail on the real data; without the volatility change the target cue of step 4 carries no direction. The gate also reports the method on two more seeds. |

## Features, models and evaluation (METHOD.md 5.3 to 5.5)

| Deviation | Reason |
| --- | --- |
| The leakage ladder uses the whole feature set A, without noise-column selection. | The ladder is defined as one fixed model on a fixed feature set; selection is part of fitting the S7 models. |
| Price-space neighbours use the trimmed, NaN-aware distance of S3 between start prices; volatility-space neighbours use the RMS difference of log RV over common stocks. Neither is standardised with fitted statistics. | Differences of log values cancel stock levels, so no statistic has to be fitted on training rows. |
| Neighbour and lag features also come as ratios to the current RV (`*_r`). | The target is a ratio to the current RV; trees cannot build ratios of two inputs. |
| LightGBM: after early stopping on the inner split, the model is refitted on all training rows with that number of trees. The MLP does the same with its number of epochs. | Otherwise the most recent 15% of training time_ids, the ones next to the validation block, would never be trained on. |
| Blend weights of fold k are fitted on the out-of-fold predictions of earlier folds only; the first fold uses equal weights. | "Out-of-fold predictions only" read in the strict, time-respecting sense. |
| Ablation uses seed 0 only and is compared with the seed-0 prediction of the full model. | Nine extra model fits per seed; stated with every ablation number. |
| Diebold-Mariano: Bartlett lag = floor(T^(1/3)), Harvey-Leybourne-Newbold correction with h = lag + 1. | The protocol names the estimators but not the lag. |
| `results/model_table.csv` holds the runs listed under `runs` in `configs/base.yaml`, not every (model, feature set, scheme) combination. | The leakage ladder needs four schemes for one model; the model ladder needs one scheme. |
| Figures are produced by `python -m lobvol evaluate` (`report.py`); there are no notebooks. | Every number and figure is regenerated by the CLI. |

## Holdout (METHOD.md 5.6)

| Deviation | Reason |
| --- | --- |
| Step 1: no commit was frozen. `results/holdout/freeze.json` records the last commit and a SHA-256 of the package sources and configs at the time of the run. | The code was committed only after the full run. |

## Corrections after the holdout

| Correction | Effect |
| --- | --- |
| The training time of the first LightGBM run (feature set A) included the one-off Optuna study, because tuning started lazily inside the timed call. Tuning now runs before the timer starts, and that run was repeated. | Training time only; predictions and every RMSPE are unchanged. |
