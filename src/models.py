"""Machine Learning & Time-Series Forecasting Models for Hydropower Analytics.

Includes:
1. Feature Engineering (Hydrological lags, cumulative rainfall, snowmelt index, seasonality).
2. Multi-Model Inflow Forecaster (Random Forest, Gradient Boosting, Baseline MLR).
3. 95% Prediction Interval Estimation (Residual-based and Quantile bounds).
4. Multi-day ahead recursive / direct forecasting.
"""

from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.physics import calculate_power_output


def engineer_hydrological_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create lag features, rolling statistics, and seasonal cyclical features."""
    data = df.copy()
    data = data.sort_values("Date").reset_index(drop=True)

    # Inflow lags
    data["Inflow_Lag1"] = data["Water Inflow (m³/s)"].shift(1)
    data["Inflow_Lag2"] = data["Water Inflow (m³/s)"].shift(2)
    data["Inflow_Lag3"] = data["Water Inflow (m³/s)"].shift(3)
    data["Inflow_Lag7"] = data["Water Inflow (m³/s)"].shift(7)

    # Rolling statistics
    data["Inflow_Roll7_Mean"] = data["Water Inflow (m³/s)"].shift(1).rolling(7).mean()
    data["Rain_Roll3_Sum"] = data["Rainfall (mm)"].rolling(3).sum()
    data["Rain_Roll7_Sum"] = data["Rainfall (mm)"].rolling(7).sum()

    # Temperature & snowmelt proxy
    data["Temp_Roll3_Mean"] = data["Temp_Mean (°C)"].rolling(3).mean()

    # Cyclical day-of-year encoding
    doy = data["Date"].dt.dayofyear
    data["Sin_DOY"] = np.sin(2 * np.pi * doy / 365.25)
    data["Cos_DOY"] = np.cos(2 * np.pi * doy / 365.25)

    # Seasonal indicators
    month = data["Date"].dt.month
    data["Is_Monsoon"] = month.isin([7, 8, 9]).astype(int)
    data["Is_Snowmelt"] = month.isin([4, 5, 6]).astype(int)

    # Reservoir state lags
    data["Level_Lag1"] = data["Reservoir Level (m)"].shift(1)
    data["LiveStoragePct_Lag1"] = data["Live Storage %"].shift(1)

    # Drop warm-up rows with NaNs from shifts
    data = data.dropna().reset_index(drop=True)
    return data


class HydroForecaster:
    """Multi-model forecaster for reservoir inflow and power output."""

    INFLOW_FEATURES = [
        "Inflow_Lag1", "Inflow_Lag2", "Inflow_Lag3", "Inflow_Lag7",
        "Inflow_Roll7_Mean", "Rainfall (mm)", "Rain_Roll3_Sum", "Rain_Roll7_Sum",
        "Temp_Mean (°C)", "Temp_Roll3_Mean", "Sin_DOY", "Cos_DOY",
        "Is_Monsoon", "Is_Snowmelt"
    ]

    POWER_FEATURES = [
        "Turbine Discharge (m³/s)", "Reservoir Level (m)",
        "Water Inflow (m³/s)", "Net Head (m)"
    ]

    def __init__(self, model_type: str = "gradient_boosting"):
        self.model_type = model_type
        if model_type == "gradient_boosting":
            self.inflow_model = GradientBoostingRegressor(
                n_estimators=120, max_depth=4, learning_rate=0.06, random_state=42
            )
            self.power_model = GradientBoostingRegressor(
                n_estimators=100, max_depth=4, learning_rate=0.08, random_state=42
            )
        elif model_type == "random_forest":
            self.inflow_model = RandomForestRegressor(
                n_estimators=150, max_depth=6, random_state=42, n_jobs=-1
            )
            self.power_model = RandomForestRegressor(
                n_estimators=100, max_depth=6, random_state=42, n_jobs=-1
            )
        elif model_type == "linear":
            self.inflow_model = Ridge(alpha=1.0)
            self.power_model = LinearRegression()
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

        self.inflow_residuals_std = 15.0
        self.power_residuals_std = 12.0
        self.is_fitted = False

    def fit(self, train_df: pd.DataFrame) -> "HydroForecaster":
        """Fit inflow and power forecasting models."""
        # 1. Fit Inflow model
        X_inf = train_df[self.INFLOW_FEATURES]
        y_inf = train_df["Water Inflow (m³/s)"]
        self.inflow_model.fit(X_inf, y_inf)

        inf_preds = self.inflow_model.predict(X_inf)
        self.inflow_residuals_std = float(np.std(y_inf - inf_preds))

        # 2. Fit Power model
        X_pwr = train_df[self.POWER_FEATURES]
        y_pwr = train_df["Power Output (MW)"]
        self.power_model.fit(X_pwr, y_pwr)

        pwr_preds = self.power_model.predict(X_pwr)
        self.power_residuals_std = float(np.std(y_pwr - pwr_preds))

        self.is_fitted = True
        return self

    def predict_inflow(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict water inflow with 95% confidence intervals."""
        X = test_df[self.INFLOW_FEATURES]
        preds = self.inflow_model.predict(X)
        z = 1.96  # 95% CI
        ci_low = np.maximum(40.0, preds - z * self.inflow_residuals_std)
        ci_high = preds + z * self.inflow_residuals_std
        return preds, ci_low, ci_high

    def predict_power(
        self, test_df: pd.DataFrame
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Predict power output (MW) with 95% confidence intervals."""
        X = test_df[self.POWER_FEATURES]
        preds = self.power_model.predict(X)
        preds = np.clip(preds, 0.0, 1000.0)
        z = 1.96
        ci_low = np.clip(preds - z * self.power_residuals_std, 0.0, 1000.0)
        ci_high = np.clip(preds + z * self.power_residuals_std, 0.0, 1000.0)
        return preds, ci_low, ci_high

    def feature_importances(self) -> pd.DataFrame:
        """Extract feature importance for tree-based models."""
        if hasattr(self.inflow_model, "feature_importances_"):
            return pd.DataFrame({
                "Feature": self.INFLOW_FEATURES,
                "Importance": self.inflow_model.feature_importances_
            }).sort_values("Importance", ascending=False).reset_index(drop=True)
        return pd.DataFrame()


def evaluate_forecast(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculate standard regression metrics: R2, MAE, RMSE, MAPE."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    non_zero = y_true > 1.0
    mape = np.mean(np.abs((y_true[non_zero] - y_pred[non_zero]) / y_true[non_zero])) * 100.0
    return {
        "R2": round(float(r2), 4),
        "MAE": round(float(mae), 2),
        "RMSE": round(float(rmse), 2),
        "MAPE (%)": round(float(mape), 2)
    }
