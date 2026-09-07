"""Time-Series Backtesting and Model Comparison for Hydropower Forecasting.

Implements:
1. Out-of-Sample Temporal Train/Test Split (Train on 2023, Test on 2024).
2. Walk-Forward Expanding Window Validation (TimeSeriesSplit).
3. Benchmark comparison: Linear Regression vs Random Forest vs Gradient Boosting.
"""

from typing import Dict, List, Any
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from src.data_loader import simulate_dam_operations
from src.models import (
    HydroForecaster,
    engineer_hydrological_features,
    evaluate_forecast
)


def run_temporal_train_test(
    data_df: pd.DataFrame,
    split_date: str = "2024-01-01"
) -> Dict[str, Any]:
    """Perform temporal out-of-sample evaluation split at split_date."""
    feat_df = engineer_hydrological_features(data_df)

    train_mask = feat_df["Date"] < split_date
    test_mask = feat_df["Date"] >= split_date

    train_df = feat_df[train_mask].reset_index(drop=True)
    test_df = feat_df[test_mask].reset_index(drop=True)

    results = {}
    models_to_test = ["linear", "random_forest", "gradient_boosting"]

    for m_type in models_to_test:
        forecaster = HydroForecaster(model_type=m_type)
        forecaster.fit(train_df)

        # Inflow evaluation
        inf_preds, inf_low, inf_high = forecaster.predict_inflow(test_df)
        inf_metrics = evaluate_forecast(test_df["Water Inflow (m³/s)"].values, inf_preds)

        # Power evaluation
        pwr_preds, pwr_low, pwr_high = forecaster.predict_power(test_df)
        pwr_metrics = evaluate_forecast(test_df["Power Output (MW)"].values, pwr_preds)

        results[m_type] = {
            "forecaster": forecaster,
            "inflow_metrics": inf_metrics,
            "power_metrics": pwr_metrics,
            "inflow_predictions": inf_preds,
            "inflow_ci_low": inf_low,
            "inflow_ci_high": inf_high,
            "power_predictions": pwr_preds,
            "power_ci_low": pwr_low,
            "power_ci_high": pwr_high
        }

    return {
        "train_df": train_df,
        "test_df": test_df,
        "results": results
    }


def run_walk_forward_cv(
    data_df: pd.DataFrame,
    n_splits: int = 5
) -> pd.DataFrame:
    """Run walk-forward expanding window cross-validation using TimeSeriesSplit."""
    feat_df = engineer_hydrological_features(data_df)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    records = []
    models_to_test = ["linear", "random_forest", "gradient_boosting"]

    for fold, (train_idx, test_idx) in enumerate(tscv.split(feat_df)):
        tr = feat_df.iloc[train_idx]
        te = feat_df.iloc[test_idx]

        for m_name in models_to_test:
            fc = HydroForecaster(model_type=m_name)
            fc.fit(tr)

            # Inflow
            inf_pred, _, _ = fc.predict_inflow(te)
            inf_m = evaluate_forecast(te["Water Inflow (m³/s)"].values, inf_pred)

            # Power
            pwr_pred, _, _ = fc.predict_power(te)
            pwr_m = evaluate_forecast(te["Power Output (MW)"].values, pwr_pred)

            records.append({
                "Fold": fold + 1,
                "Model": m_name.replace("_", " ").title(),
                "Inflow_R2": inf_m["R2"],
                "Inflow_MAE": inf_m["MAE"],
                "Inflow_RMSE": inf_m["RMSE"],
                "Power_R2": pwr_m["R2"],
                "Power_MAE": pwr_m["MAE"],
                "Power_RMSE": pwr_m["RMSE"]
            })

    return pd.DataFrame(records)


if __name__ == "__main__":
    print("Simulating Tehri operations dataset...")
    df = simulate_dam_operations("2023-01-01", "2024-12-31")

    print("\nRunning Temporal Out-of-Sample Train/Test Split (Train 2023, Test 2024)...")
    eval_pack = run_temporal_train_test(df, split_date="2024-01-01")

    print("\n" + "=" * 65)
    print("OUT-OF-SAMPLE TEST RESULTS (YEAR 2024 - 366 DAYS)")
    print("=" * 65)
    for m_type, pack in eval_pack["results"].items():
        print(f"\nModel: {m_type.upper()}")
        print("  Inflow Forecast:", pack["inflow_metrics"])
        print("  Power Forecast: ", pack["power_metrics"])
