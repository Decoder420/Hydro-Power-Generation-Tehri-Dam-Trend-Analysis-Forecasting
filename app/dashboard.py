"""Streamlit Operator Dashboard for Tehri Hydroelectric Complex.

Features:
1. Executive KPIs: Reservoir elevation, live storage, power output, head, water rate.
2. Rule Curve Monitor & Flood Cushion Warning System.
3. Multi-Model Inflow & Generation Forecasting with 95% Confidence Intervals.
4. "What-If" Extreme Storm / Cloudburst Simulator.
5. 96-Time-Block (15-min) Day-Ahead Dispatch Scheduler for NRLDC submission.
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import numpy as np
import pandas as pd
import streamlit as st

from src.backtest import run_temporal_train_test
from src.data_loader import simulate_dam_operations
from src.dispatcher import DayAheadScheduler, TehriRuleCurveEngine, simulate_what_if_scenario
from src.physics import FRL, MDDL, MAX_CAPACITY_MW, MAX_DISCHARGE, TEHRI_SPECS

# Page configuration
st.set_page_config(
    page_title="Tehri Hydroelectric Dispatch & Forecast Console",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Government of India Styling
st.markdown("""
<style>
    .gov-tricolor {
        height: 4px;
        background: linear-gradient(90deg, #ff9933 0%, #ff9933 33.3%, #ffffff 33.3%, #ffffff 66.6%, #138808 66.6%, #138808 100%);
        width: 100%;
        margin-bottom: 8px;
    }
    .gov-header-box {
        background: linear-gradient(135deg, #0b2545 0%, #133b5c 100%);
        color: #ffffff;
        padding: 16px 20px;
        border-radius: 6px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(11,37,69,0.15);
    }
    .gov-sup-title {
        font-size: 0.8rem;
        color: #f3e5ab;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .gov-main-title {
        font-size: 1.8rem;
        font-weight: 800;
        color: #ffffff;
        margin: 2px 0;
    }
    .gov-sub-title {
        font-size: 0.88rem;
        color: #94a3b8;
    }
    .kpi-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 14px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
        border-top: 3px solid #0b2545;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 12px;
        font-size: 0.85rem;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data(show_spinner="Simulating Tehri operational physics & weather feeds...")
def load_cached_data():
    """Load and cache full operational history and model backtest."""
    df = simulate_dam_operations(start_date="2023-01-01", end_date="2024-12-31")
    eval_pack = run_temporal_train_test(df, split_date="2024-01-01")
    return df, eval_pack


df, eval_pack = load_cached_data()

# Tricolor stripe and Government Header
st.markdown('<div class="gov-tricolor"></div>', unsafe_allow_html=True)
st.markdown("""
<div class="gov-header-box">
  <div class="gov-sup-title">भारत सरकार | Government of India • विद्युत मंत्रालय | Ministry of Power</div>
  <div class="gov-main-title">केंद्रीय विद्युत प्राधिकरण | Central Electricity Authority & THDC India Limited</div>
  <div class="gov-sub-title">
    राष्ट्रीय जलविद्युत प्रचालन एवं ग्रिड प्रेषण पोर्टल (NHOP-DSS) • <strong>Tehri Hydro Power Complex (1,000 MW HPP Stage-I)</strong>
  </div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/6/6b/Tehri_dam.jpg/640px-Tehri_dam.jpg", use_container_width=True, caption="Tehri Dam (260.5 m Earth & Rockfill)")
st.sidebar.title("Operational Controls")

# Date selector for inspection
latest_date = df["Date"].max()
default_date = df["Date"].iloc[-15]
selected_date = st.sidebar.date_input("Select Operating Date", value=default_date, min_value=df["Date"].min(), max_value=latest_date)
selected_ts = pd.to_datetime(selected_date)

# Fetch slice
row = df[df["Date"] == selected_ts]
if row.empty:
    row = df.iloc[-1]
else:
    row = row.iloc[0]

# Evaluate reservoir status
status_info = TehriRuleCurveEngine.evaluate_reservoir_status(row["Reservoir Level (m)"], row["Date"].month)

# Top KPI Metric Row
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="Reservoir Elevation",
        value=f"{row['Reservoir Level (m)']:.2f} m",
        delta=f"FRL: {FRL}m | MDDL: {MDDL}m",
        delta_color="off"
    )
with kpi2:
    st.metric(
        label="Live Storage Capacity",
        value=f"{row['Live Storage %']:.1f} %",
        delta=f"{row['Storage (MCM)']:.0f} / 3540 MCM",
        delta_color="off"
    )
with kpi3:
    st.metric(
        label="Power Output",
        value=f"{row['Power Output (MW)']:.1f} MW",
        delta=f"Cap: {MAX_CAPACITY_MW} MW ({100 * row['Power Output (MW)']/MAX_CAPACITY_MW:.1f}%)"
    )
with kpi4:
    st.metric(
        label="Water Inflow",
        value=f"{row['Water Inflow (m³/s)']:.1f} m³/s",
        delta=f"Discharge: {row['Turbine Discharge (m³/s)']:.1f} m³/s"
    )
with kpi5:
    st.metric(
        label="Water Rate Efficiency",
        value=f"{row['Water Rate (m³/MWh)']:.0f} m³/MWh",
        delta=f"Net Head: {row['Net Head (m)']:.1f} m"
    )

# Rule Curve Status Banner
if status_info["severity"] == "danger":
    st.error(f"**{status_info['status']}** — {status_info['recommendation']}")
elif status_info["severity"] == "warning":
    st.warning(f"**{status_info['status']}** — {status_info['recommendation']}")
else:
    st.success(f"**Rule Curve Status: {status_info['status']}** — {status_info['recommendation']}")

# Navigation Tabs
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Operational Trends & Mass Balance",
    "Inflow & Generation Forecasting",
    "'What-If' Storm Simulator",
    "Day-Ahead 96-Block Dispatch",
    "Plant Specifications & Methodology"
])

with tab1:
    st.subheader("Hydrological Mass Balance & Generation History")
    col_a, col_b = st.columns(2)

    with col_a:
        st.write("#### Water Inflow vs Turbine Discharge")
        trend_df = df[["Date", "Water Inflow (m³/s)", "Turbine Discharge (m³/s)"]].set_index("Date")
        st.line_chart(trend_df, color=["#1f77b4", "#2ca02c"])

    with col_b:
        st.write("#### Power Generation (MW) & Reservoir Elevation (m)")
        gen_df = df[["Date", "Power Output (MW)"]].set_index("Date")
        st.area_chart(gen_df, color="#ff7f0e")

    col_c, col_d = st.columns(2)
    with col_c:
        st.write("#### Reservoir Elevation vs Full Reservoir Level (830m)")
        res_df = df[["Date", "Reservoir Level (m)"]].set_index("Date")
        st.line_chart(res_df, color="#0984e3")
    with col_d:
        st.write("#### Water Rate Efficiency (m³ of water consumed per MWh)")
        wr_df = df[["Date", "Water Rate (m³/MWh)"]].set_index("Date")
        st.line_chart(wr_df, color="#8e44ad")

with tab2:
    st.subheader("Machine Learning Inflow & Generation Forecasting Engine")
    st.markdown(
        "Trained on historical catchment meteorology & antecedent hydrological states with strictly out-of-sample temporal validation."
    )

    test_df = eval_pack["test_df"]
    best_results = eval_pack["results"]["gradient_boosting"]

    f_col1, f_col2 = st.columns([2, 1])

    with f_col1:
        st.write("#### 2024 Out-of-Sample Forecast vs Actual Inflow")
        forecast_plot_df = pd.DataFrame({
            "Date": test_df["Date"],
            "Actual Inflow (m³/s)": test_df["Water Inflow (m³/s)"],
            "Predicted Inflow (m³/s)": best_results["inflow_predictions"]
        }).set_index("Date")
        st.line_chart(forecast_plot_df, color=["#34495e", "#e74c3c"])

    with f_col2:
        st.write("#### Out-of-Sample Evaluation Metrics")
        metrics_df = pd.DataFrame({
            "Metric": ["R² Score", "Mean Absolute Error (MAE)", "Root Mean Squared Error (RMSE)", "Mean Absolute % Error (MAPE)"],
            "Inflow Model": [
                f"{best_results['inflow_metrics']['R2']:.4f}",
                f"{best_results['inflow_metrics']['MAE']:.2f} m³/s",
                f"{best_results['inflow_metrics']['RMSE']:.2f} m³/s",
                f"{best_results['inflow_metrics']['MAPE (%)']:.2f} %"
            ],
            "Power Model": [
                f"{best_results['power_metrics']['R2']:.4f}",
                f"{best_results['power_metrics']['MAE']:.2f} MW",
                f"{best_results['power_metrics']['RMSE']:.2f} MW",
                f"{best_results['power_metrics']['MAPE (%)']:.2f} %"
            ]
        })
        st.table(metrics_df)

        st.info("**Physics-Informed Features:** Inflow Lags (1, 2, 7 days), 3-day and 7-day cumulative rainfall, snowmelt degree-day indicator, and net hydraulic head.")

    st.write("#### Top Hydrological Feature Importances")
    feat_imp = best_results["forecaster"].feature_importances()
    if not feat_imp.empty:
        st.bar_chart(feat_imp.set_index("Feature").head(8))

with tab3:
    st.subheader("Extreme Hydrological Event & 'What-If' Storm Simulator")
    st.markdown("Test dam response to sudden cloudbursts, monsoon storms, or dry spells to evaluate flood buffer security and spillway risk.")

    sc_col1, sc_col2 = st.columns(2)
    with sc_col1:
        rain_input = st.slider("Catchment Storm Rainfall Event (mm total over 4 days)", min_value=0, max_value=200, value=75, step=5)
        turb_q_input = st.slider("Planned Turbine Release (m³/s)", min_value=50, max_value=500, value=350, step=25)
    with sc_col2:
        initial_elev = st.slider("Current Starting Reservoir Level (m MSL)", min_value=750.0, max_value=830.0, value=float(row["Reservoir Level (m)"]), step=0.5)
        horizon = st.slider("Simulation Horizon (Days)", min_value=3, max_value=14, value=7)

    sim_res = simulate_what_if_scenario(
        current_elevation=initial_elev,
        rainfall_scenario_mm=rain_input,
        horizon_days=horizon,
        daily_turbine_q=turb_q_input
    )

    st.write("#### 7-Day Projected Reservoir Trajectory")
    st.dataframe(sim_res, use_container_width=True)

    peak_elev = sim_res["Reservoir_Elevation_m"].max()
    total_spill = sim_res["Spillway_Q_m3_s"].sum()
    total_gwh = sim_res["Daily_Energy_GWh"].sum()

    m1, m2, m3 = st.columns(3)
    m1.metric("Peak Projected Elevation", f"{peak_elev:.2f} m", delta=f"{peak_elev - FRL:.2f} m vs FRL" if peak_elev > FRL else f"{FRL - peak_elev:.2f} m margin below FRL")
    m2.metric("Spillway Overflow Required", f"{total_spill:.1f} m³/s", delta="Safe (No Spilling)" if total_spill == 0 else "Spillway Active!", delta_color="inverse" if total_spill > 0 else "normal")
    m3.metric("Projected Energy Generation", f"{total_gwh:.2f} GWh", delta=f"~{total_gwh * 1000 / (horizon * 24):.0f} MW avg")

with tab4:
    st.subheader("Day-Ahead 96-Time-Block Generation Scheduler (NRLDC Format)")
    st.markdown(
        "Generates 15-minute generation declaration curves tailored to Northern Indian Grid peaking profiles with Deviation Settlement Mechanism (DSM) $\\pm 12\\%$ tolerance bands."
    )

    sched_col1, sched_col2 = st.columns(2)
    with sched_col1:
        target_energy = st.number_input("Target Daily Energy Commitment (MWh)", min_value=2000.0, max_value=24000.0, value=8800.0, step=200.0)
    with sched_col2:
        sched_date = st.date_input("Schedule Date", value=selected_date)

    schedule_df = DayAheadScheduler.generate_96_block_schedule(
        date_str=str(sched_date),
        reservoir_level=row["Reservoir Level (m)"],
        daily_target_energy_mwh=target_energy,
        inflow_m3_per_s=row["Water Inflow (m³/s)"]
    )

    st.line_chart(schedule_df[["Time", "Scheduled_MW", "DSM_Band_Low_MW", "DSM_Band_High_MW"]].set_index("Time"))

    # Download button
    csv_data = schedule_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download 96-Block Dispatch Schedule (CSV for RLDC)",
        data=csv_data,
        file_name=f"Tehri_96Block_Schedule_{sched_date}.csv",
        mime="text/csv"
    )

    with st.expander("View Full 96-Time-Block Schedule Table"):
        st.dataframe(schedule_df, use_container_width=True)

with tab5:
    st.subheader("Tehri Hydroelectric Complex Technical Specifications")
    st.json(TEHRI_SPECS)
