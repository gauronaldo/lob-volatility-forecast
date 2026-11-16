Holdout: 766 time_ids, 85,792 rows; every model refitted on the full development set; naive RMSPE on the same rows 0.3299.

| model | features | rmspe | interval | seeds | train_seconds |
| --- | --- | --- | --- | --- | --- |
| Naive (RV of the last 10 min) | A | 0.3299 | 0.3178 to 0.3435 | 0 | 0 |
| HAR ridge | A | 0.2426 | 0.2352 to 0.2499 | 0 | 1 |
| Random forest | D | 0.2162 | 0.2091 to 0.2232 | 0 | 105 |
| LightGBM | A | 0.2262 | 0.2194 to 0.2325 | 0 1 2 | 75 |
| LightGBM | B | 0.2266 | 0.2198 to 0.2330 | 0 1 2 | 56 |
| LightGBM | C_strict | 0.2155 | 0.2087 to 0.2226 | 0 1 2 | 20 |
| LightGBM | C_comp | 0.2075 | 0.2008 to 0.2143 | 0 1 2 | 17 |
| LightGBM | C_strict_gap | 0.2194 | 0.2123 to 0.2266 | 0 1 2 | 18 |
| LightGBM | D | 0.2129 | 0.2064 to 0.2199 | 0 1 2 | 20 |
| LightGBM | D_gap | 0.2186 | 0.2113 to 0.2257 | 0 1 2 | 19 |
| MLP | D | 0.2104 | 0.2035 to 0.2175 | 0 1 2 | 245 |
| Blend | D | 0.2103 | 0.2034 to 0.2173 | - | 0 |

Paired differences on the holdout (95% block-bootstrap intervals):

| comparison | rmspe_a | rmspe_b | difference | ci_low | ci_high |
| --- | --- | --- | --- | --- | --- |
| RQ3 look-ahead: C_strict minus C_comp | 0.2155 | 0.2075 | +0.0081 | +0.0064 | +0.0096 |
| RQ1: LightGBM A minus naive | 0.2262 | 0.3299 | -0.1038 | -0.1141 | -0.0949 |
| RQ1: B minus A | 0.2266 | 0.2262 | +0.0005 | +0.0002 | +0.0007 |
| RQ1: C_strict minus B | 0.2155 | 0.2266 | -0.0111 | -0.0138 | -0.0084 |
| RQ1: D minus C_strict | 0.2129 | 0.2155 | -0.0026 | -0.0036 | -0.0018 |
| RQ1: D minus B | 0.2129 | 0.2266 | -0.0137 | -0.0166 | -0.0109 |
| Order error: C_strict_gap minus C_strict | 0.2194 | 0.2155 | +0.0038 | +0.0024 | +0.0051 |
| Order error: C_strict_gap minus B | 0.2194 | 0.2266 | -0.0073 | -0.0104 | -0.0044 |
| Order error: D_gap minus D | 0.2186 | 0.2129 | +0.0056 | +0.0040 | +0.0071 |
| Order error: D_gap minus B | 0.2186 | 0.2266 | -0.0081 | -0.0114 | -0.0051 |
| Model: HAR ridge minus naive | 0.2426 | 0.3299 | -0.0874 | -0.0967 | -0.0796 |
| Model: forest minus HAR ridge | 0.2162 | 0.2426 | -0.0264 | -0.0289 | -0.0233 |
| Model: LightGBM D minus forest | 0.2129 | 0.2162 | -0.0033 | -0.0041 | -0.0024 |
| RQ4: MLP minus LightGBM D | 0.2104 | 0.2129 | -0.0025 | -0.0035 | -0.0016 |
| Model: blend minus LightGBM D | 0.2103 | 0.2129 | -0.0026 | -0.0034 | -0.0019 |

Diebold-Mariano tests on adjacent rungs (negative statistic: the model beats the rung below it):

| ladder | model | against | statistic | p_value | lag |
| --- | --- | --- | --- | --- | --- |
| features | gbdt__A | naive__A | -17.85 | 0.0000 | 9 |
| features | gbdt__B | gbdt__A | +3.51 | 0.0005 | 9 |
| features | gbdt__C_strict | gbdt__B | -9.50 | 0.0000 | 9 |
| features | gbdt__D | gbdt__C_strict | -5.74 | 0.0000 | 9 |
| models | har_ridge__A | naive__A | -17.11 | 0.0000 | 9 |
| models | forest__D | har_ridge__A | -18.46 | 0.0000 | 9 |
| models | gbdt__D | forest__D | -6.77 | 0.0000 | 9 |
| models | mlp__D | gbdt__D | -5.05 | 0.0000 | 9 |
| models | blend__D | mlp__D | -0.85 | 0.3982 | 9 |