# Handoff

State of the project as of 2026-11-17, for whoever picks it up next.

## Status

Every stage of [METHOD.md](METHOD.md) is done, S1 audit to S9 report, and
every gate passed. The holdout was opened once, on 2026-11-16, after all
models were frozen. **Do not run it again.** `python -m lobvol holdout`
refuses a second run, and the protocol allows no modelling change after it.

Results are in [README.md](../README.md) (summary) and
[reports/report.md](../reports/report.md) (everything else). Every number in
both is written by `python -m lobvol evaluate` into the
`<!-- BEGIN:name -->` blocks; do not edit numbers by hand.

## How the code is organised

- One CLI, `python -m lobvol <command>`, one command per stage; each reads
  `configs/base.yaml`, skips outputs that exist unless `--force`, and writes a
  run record.
- Derived data lives in `data/interim/` and `data/features/` (git-ignored,
  about 1.7 GB of raw data plus caches). Committed outputs live in
  `results/`.
- The folds are generated once in S4 (`cv.py`) and loaded by every model.
  `holdout.py` is the only module that reads holdout targets.
- Feature selection lists are cached per (feature set, scheme, fold), and the
  LightGBM hyperparameters in `results/gbdt_tuned.json`. Deleting that file
  reruns the 40-trial Optuna study, about 45 minutes.

## Open items

1. **Report text.** Three findings from the review are not yet in
   `reports/report.md`:
   - the gain of past neighbours over B is about half as large on the holdout
     as in development, and the reason is not known;
   - forecasts sit 8-11% below realized volatility in every decile, which is
     expected from an RMSPE loss but should be stated;
   - stock 31 has by far the largest error (RMSPE about 0.74, naive about
     1.22), while the error analysis drops stock 33, the least liquid by book
     updates. A "without stock 31" line would be more informative.
2. **Clean rebuild not verified.** The stages were run one by one. A full
   `python -m lobvol all` from a fresh clone plus the raw data (about two
   hours) has not been done.
3. **CI on GitHub not checked.** It passes locally (79 tests, ruff clean) on
   Windows; the Ubuntu job has not been observed.
4. **`results/holdout/freeze.json`.** Its source hash no longer matches the
   code: docstrings and the training-time fix changed after the holdout run.
   No prediction changed; see the last section of
   [DEVIATIONS.md](DEVIATIONS.md).
5. **SHAP.** `shap` fails to import with NumPy 2.4 (its `numba` dependency),
   so values come from LightGBM's own TreeSHAP. Revisit when `numba`
   supports NumPy 2.4.
6. **No LICENSE** has been chosen.

Any change after the holdout that touches a number needs a dated entry in
[DEVIATIONS.md](DEVIATIONS.md) with the old and new values.

## Environment notes

- Windows: pytest may lack permission on the default temp folder; pass
  `--basetemp <folder>`. Call tools as `python -m ruff` and
  `python -m pytest` if they are not on `PATH`.
- No GPU was used. The MLP accounts for most of the training time.
