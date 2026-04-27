
Baseline Model Metric

Metric: Value | Interpretation 

MAE: 7.71 | On average, our forecast misses by about 7 trips per zone-hour, so dispatch should expect roughly that level of normal over/under staffing need each hour. |

MAPE: 69.09 | For nonzero-demand hours, the average percent error is very large, which means percentage-based accuracy is unstable and route staffing can still be proportionally far off in many hours. |

MBE: 0.768 | The model has a small positive bias, so it tends to over-forecast by about 1 trip per zone-hour, which may cause slight overstaffing if used directly. |

R^2: 0.95 | The model explains about 95% of demand variation, so it captures most patterns needed for planning shifts and vehicle coverage. |

RMSE: 17.8 | Because RMSE penalizes big misses, this value suggests some hours still have large forecast errors that can impact peak-time driver allocation. |



## MAPE zero-demand handling

MAPE was computed only on rows where actual demand was greater than zero, because dividing by zero-demand hours is mathematically undefined and would produce invalid percentage errors.
For the baseline validation run, excluded zero-demand rows were 0 (0.0%), so all validation rows were included in the reported MAPE.

## Part 2 - Hyperparameter tuning results

Metric comparison:
Metric |  Baseline RF |  Tuned RF Try1 | Tuned RF Try2
-------------------------------------------------------
val_mae |    7.71     |	   16.96       |   7.668
val_rmse|    17.8     |	   32.1        |   17.38
val_r2	|    0.95     |    0.83        |   0.952
val_mape|	 69.09    |    648.7       |   73.17
val_mbe	|    0.768	  |    5.357       |   0.9



## Why Tuned RF Try2 

I included Tuned RF Try2 because the first tuning attempt (Try1) performed much worse than the baseline, so a second tuning run was necessary to verify whether that result was a one-off outcome from the initial search settings or a true indication that tuning was not helping. Try2 acts as a validation rerun: it checks reproducibility, gives a fairer comparison against the baseline, and supports a stronger final conclusion using multiple tuning attempts instead of a single outlier run.

## Best model
The best model from run_training.py was random_forest_regressor with an R^2 of 0.95. The second closest model was gradient_boosting_regressor with an R^2 of 0.945.