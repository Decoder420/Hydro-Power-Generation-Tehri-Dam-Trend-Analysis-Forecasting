# hydro_power_analysis.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

# ---------------------------------------------------------
# STEP 1: Generate Synthetic Dataset
# ---------------------------------------------------------
np.random.seed(42)
dates = pd.date_range(start="2024-01-01", periods=365, freq='D')

inflow = np.random.normal(1200, 50, 365)
inflow[180:270] += np.random.normal(300, 50, 90)  # monsoon boost

rainfall = np.random.normal(10, 5, 365)
rainfall[180:270] += np.random.normal(40, 10, 90)

reservoir_level = 300 + (inflow - 1100)/50 + np.random.normal(0, 1, 365)
power_output = inflow * 0.35 + rainfall * 2 + np.random.normal(0, 20, 365)

df = pd.DataFrame({
    "Date": dates,
    "Water Inflow (m³/s)": inflow.round(2),
    "Rainfall (mm)": rainfall.round(2),
    "Reservoir Level (m)": reservoir_level.round(2),
    "Power Output (MW)": power_output.round(2)
})

df.to_excel("THDC_Power_Analytics.xlsx", index=False)
print("Dataset saved as THDC_Power_Analytics.xlsx")

# ---------------------------------------------------------
# STEP 2: Line Chart (Trend)
# ---------------------------------------------------------
plt.figure(figsize=(12, 6))
plt.plot(df["Date"], df["Water Inflow (m³/s)"], label="Water Inflow (m³/s)", color="blue")
plt.plot(df["Date"], df["Power Output (MW)"], label="Power Output (MW)", color="green")
plt.xlabel("Date")
plt.ylabel("Value")
plt.title("Water Inflow vs Power Output (2024) - Simulated Data")
plt.legend()
plt.tight_layout()
plt.savefig("THDC_Power_Trend.png")
plt.close()

# ---------------------------------------------------------
# STEP 3: Regression Analysis (Inflow vs Power)
# ---------------------------------------------------------
X_inflow = df[["Water Inflow (m³/s)"]].values
y_power = df["Power Output (MW)"].values
lin_inflow = LinearRegression().fit(X_inflow, y_power)
y_pred_inflow = lin_inflow.predict(X_inflow)

plt.figure(figsize=(8, 6))
plt.scatter(df["Water Inflow (m³/s)"], df["Power Output (MW)"], s=10, alpha=0.5, label="Daily data")
order = np.argsort(X_inflow.flatten())
plt.plot(X_inflow.flatten()[order], y_pred_inflow[order], color="red", label="Regression line")
plt.xlabel("Water Inflow (m³/s)")
plt.ylabel("Power Output (MW)")
plt.title("Inflow vs Power Output (with Regression Line)")
plt.legend()
plt.tight_layout()
plt.savefig("Inflow_vs_Power_Regression.png")
plt.close()

# ---------------------------------------------------------
# STEP 4: Multiple Linear Regression (Inflow + Rainfall + Reservoir)
# ---------------------------------------------------------
features = df[["Water Inflow (m³/s)", "Rainfall (mm)", "Reservoir Level (m)"]]
target = df["Power Output (MW)"]

mlr = LinearRegression().fit(features, target)
pred = mlr.predict(features)

r2 = r2_score(target, pred)
mae = mean_absolute_error(target, pred)
print(f"Model Performance: R² = {r2:.3f}, MAE = {mae:.2f} MW")

# ---------------------------------------------------------
# STEP 5: Monthly Trends & Forecasting
# ---------------------------------------------------------
df["Month"] = df["Date"].dt.month
monthly_avg = df.groupby("Month").mean(numeric_only=True).reset_index()

plt.figure(figsize=(10, 5))
plt.plot(monthly_avg["Month"], monthly_avg["Power Output (MW)"], marker="o")
plt.xticks(range(1, 13))
plt.xlabel("Month")
plt.ylabel("Average Power Output (MW)")
plt.title("Monthly Average Power Output (2024)")
plt.tight_layout()
plt.savefig("Monthly_Avg_Power.png")
plt.close()

# Forecast for next January
jan_row = monthly_avg[monthly_avg["Month"] == 1].iloc[0]
X_next = np.array([[jan_row["Water Inflow (m³/s)"], jan_row["Rainfall (mm)"], jan_row["Reservoir Level (m)"]]])
next_month_avg_power = float(mlr.predict(X_next)[0])
print(f"Forecast (January): {next_month_avg_power:.2f} MW")

# Save forecast summary
summary = pd.DataFrame({
    "Metric": ["Model R2", "MAE (MW)", "Forecast (January Avg Power MW)"],
    "Value": [r2, mae, next_month_avg_power]
})
summary.to_csv("Forecast_Summary.csv", index=False)
print("Forecast Summary saved as Forecast_Summary.csv")

print("All visualizations & reports generated.")
