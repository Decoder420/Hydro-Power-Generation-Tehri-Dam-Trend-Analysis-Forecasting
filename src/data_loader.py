"""Data Ingestion, Hydrological Rainfall-Runoff Modeling, and Reservoir Mass-Balance Simulation.

Features:
1. Ingests real historical daily weather (precipitation, temperature, evapotranspiration)
   for Tehri Dam coordinates (30.3782 N, 78.4803 E) via Open-Meteo Historical Weather API.
2. Models Himalayan snowmelt (degree-day method) and catchment rainfall-runoff lag.
3. Implements daily reservoir mass-balance conservation and operational dispatch rules.
4. Computes exact physical turbine power (MW), hydraulic head, and water rate (m^3/MWh).
"""

import json
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd
import requests

from src.physics import (
    calculate_power_output,
    calculate_water_rate,
    elevation_to_storage_mcm,
    storage_to_elevation_meters,
    FRL,
    MDDL,
    MAX_DISCHARGE,
    LIVE_STORAGE_MCM,
    DEAD_STORAGE_MCM
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

CACHE_DIR = Path(__file__).resolve().parent.parent / "data"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
CACHE_FILE = CACHE_DIR / "tehri_weather_cache.parquet"


def fetch_tehri_weather(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31",
    use_cache: bool = True
) -> pd.DataFrame:
    """Fetch real historical daily meteorology for Tehri Dam catchment from Open-Meteo API.

    Falls back to calibrated seasonal data if offline or network failure occurs.
    """
    if use_cache and CACHE_FILE.exists():
        try:
            df = pd.read_parquet(CACHE_FILE)
            logger.info("Loaded cached weather data (%d records) from %s", len(df), CACHE_FILE.name)
            return df
        except Exception as e:
            logger.warning("Failed to read cache file: %s", e)

    url = (
        "https://archive-api.open-meteo.com/v1/archive?"
        "latitude=30.3782&longitude=78.4803&"
        f"start_date={start_date}&end_date={end_date}&"
        "daily=precipitation_sum,temperature_2m_mean,temperature_2m_max,temperature_2m_min,et0_fao_evapotranspiration&"
        "timezone=Asia%2FKolkata"
    )

    try:
        logger.info("Fetching real Tehri catchment weather from Open-Meteo API (%s to %s)...", start_date, end_date)
        resp = requests.get(url, timeout=12)
        resp.raise_for_status()
        data = resp.json().get("daily", {})

        df = pd.DataFrame({
            "Date": pd.to_datetime(data["time"]),
            "Rainfall (mm)": np.array(data.get("precipitation_sum", []), dtype=float),
            "Temp_Mean (C)": np.array(data.get("temperature_2m_mean", []), dtype=float),
            "Temp_Max (C)": np.array(data.get("temperature_2m_max", []), dtype=float),
            "Temp_Min (C)": np.array(data.get("temperature_2m_min", []), dtype=float),
            "Evaporation (mm)": np.array(data.get("et0_fao_evapotranspiration", []), dtype=float)
        })

        # Impute any missing values
        df = df.bfill().ffill()

        # Cache locally
        df.to_parquet(CACHE_FILE, index=False)
        logger.info("Successfully fetched and cached %d daily weather records.", len(df))
        return df

    except Exception as exc:
        logger.warning("Open-Meteo API request failed (%s). Generating calibrated Tehri climate data.", exc)
        return _generate_calibrated_weather(start_date, end_date)


def _generate_calibrated_weather(start_date: str, end_date: str) -> pd.DataFrame:
    """Generate realistic climate series matching IMD Tehri Garhwal normals if offline."""
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n = len(dates)
    day_of_year = dates.dayofyear.values

    # Seasonal temperature cycle (peak in May/June ~ 28C, winter ~ 8C)
    temp_mean = 18.0 + 10.0 * np.sin(2 * np.pi * (day_of_year - 105) / 365.25) + np.random.normal(0, 1.8, n)

    # Monsoon rainfall (peaks mid-July to end of August, DOY 180 to 260)
    monsoon_prob = np.exp(-0.5 * ((day_of_year - 215) / 30) ** 2)
    rain = np.where(
        np.random.rand(n) < (0.15 + 0.65 * monsoon_prob),
        np.random.exponential(scale=5.0 + 35.0 * monsoon_prob),
        0.0
    )

    evap = np.clip(1.5 + 0.12 * temp_mean + np.random.normal(0, 0.3, n), 0.5, 7.0)

    return pd.DataFrame({
        "Date": dates,
        "Rainfall (mm)": np.round(rain, 2),
        "Temp_Mean (C)": np.round(temp_mean, 1),
        "Temp_Max (C)": np.round(temp_mean + 5.0, 1),
        "Temp_Min (C)": np.round(temp_mean - 5.0, 1),
        "Evaporation (mm)": np.round(evap, 2)
    })


def simulate_catchment_inflow(weather_df: pd.DataFrame) -> pd.Series:
    """Simulate river inflow (m^3/s) to Tehri reservoir from weather drivers.

    Hydrological mechanisms:
    1. Baseflow: Perennial Himalayan glacial/spring baseflow (~75-120 m^3/s).
    2. Snowmelt: Temperature degree-day melt during April-June before monsoon.
    3. Runoff: Rainfall converted to runoff through a multi-day catchment unit hydrograph.
    """
    rain = weather_df["Rainfall (mm)"].values
    temp = weather_df["Temp_Mean (C)"].values
    doy = weather_df["Date"].dt.dayofyear.values
    n = len(weather_df)

    # 1. Baseflow (slow groundwater drainage)
    baseflow = 90.0 + 20.0 * np.sin(2 * np.pi * (doy - 120) / 365.25)

    # 2. Himalayan Snowmelt (Gangotri & high-altitude glaciers active April-June)
    is_snowmelt = (doy >= 100) & (doy <= 190)
    snowmelt = np.where(is_snowmelt, np.clip((temp - 12.0) * 18.0, 0.0, 320.0), 0.0)

    # 3. Catchment rainfall runoff with 3-day hydrological response kernel
    # Tehri catchment area ~ 7,511 km^2; mountainous orographic rainfall multiplier
    monsoon_factor = np.where((doy >= 170) & (doy <= 270), 32.0, 14.0)
    runoff_raw = rain * monsoon_factor
    runoff = np.zeros(n)
    for i in range(n):
        r0 = runoff_raw[i] * 0.35
        r1 = runoff_raw[i - 1] * 0.45 if i >= 1 else runoff_raw[i] * 0.45
        r2 = runoff_raw[i - 2] * 0.20 if i >= 2 else runoff_raw[i] * 0.20
        runoff[i] = r0 + r1 + r2

    # Total river inflow (m^3/s)
    total_inflow = baseflow + snowmelt + runoff + np.random.normal(0, 10.0, n)
    total_inflow = np.maximum(50.0, total_inflow)

    return pd.Series(np.round(total_inflow, 2), name="Water Inflow (m³/s)")


def simulate_dam_operations(
    start_date: str = "2023-01-01",
    end_date: str = "2024-12-31"
) -> pd.DataFrame:
    """Run full physical simulation of Tehri Dam operations and reservoir mass-balance.

    Water balance:
        S[t] = S[t-1] + (Inflow[t] - Discharge[t] - Spill[t] - Evap[t]) * dt
    """
    weather_df = fetch_tehri_weather(start_date=start_date, end_date=end_date)
    weather_df["Water Inflow (m³/s)"] = simulate_catchment_inflow(weather_df)

    n = len(weather_df)
    dates = weather_df["Date"]
    inflow = weather_df["Water Inflow (m³/s)"].values

    # State variables
    res_storage = np.zeros(n)
    res_elevation = np.zeros(n)
    turbine_discharge = np.zeros(n)
    spill_discharge = np.zeros(n)

    # Initial condition: Jan 1 starts at typical winter elevation ~820 m
    initial_storage = elevation_to_storage_mcm(820.0)
    current_storage = initial_storage

    # Conversion factor: 1 m^3/s for 1 day = 86400 m^3 = 0.0864 MCM
    M3_S_TO_MCM_DAY = 0.0864

    for t in range(n):
        doy = dates.iloc[t].dayofyear
        current_elev = storage_to_elevation_meters(current_storage)
        inf_today = inflow[t]

        # Operational Dispatch Strategy:
        # 1. Non-monsoon drafting (DOY 1 to 170): Controlled peaking release ~110-180 m^3/s, lowering to ~765 m
        # 2. Monsoon filling (DOY 171 to 270): Run turbines at full capacity (400-500 m^3/s); surplus fills reservoir to 825-830 m
        # 3. Post-monsoon conservation (DOY 271 to 365): High head operation (~825 m), firm peaking release ~120-170 m^3/s
        if doy < 170:
            target_q = min(110.0 + (doy / 170.0) * 60.0 + (inf_today * 0.15), 240.0)
        elif 170 <= doy <= 270:
            target_q = min(380.0 + (inf_today * 0.12), MAX_DISCHARGE)
        else:
            target_q = min(120.0 + (inf_today * 0.20), 220.0)

        # Environmental flow requirement: minimum 40 m^3/s released downstream
        actual_turb_q = max(40.0, min(target_q, MAX_DISCHARGE))

        # Check if reservoir would drop below MDDL (740m)
        min_storage = DEAD_STORAGE_MCM + 10.0
        max_possible_outflow_mcm = max(0.0, current_storage + (inf_today * M3_S_TO_MCM_DAY) - min_storage)
        actual_turb_q = min(actual_turb_q, max_possible_outflow_mcm / M3_S_TO_MCM_DAY)

        # Reservoir mass balance step
        inflow_mcm = inf_today * M3_S_TO_MCM_DAY
        turb_mcm = actual_turb_q * M3_S_TO_MCM_DAY
        evap_mcm = 0.45  # Daily evaporation losses (~0.45 MCM/day average)

        net_storage = current_storage + inflow_mcm - turb_mcm - evap_mcm

        # Check spillway activation (if storage exceeds FRL 830m storage of 3540 MCM)
        max_storage = elevation_to_storage_mcm(FRL)
        if net_storage > max_storage:
            excess_mcm = net_storage - max_storage
            actual_spill_q = excess_mcm / M3_S_TO_MCM_DAY
            net_storage = max_storage
        else:
            actual_spill_q = 0.0

        # Save states
        current_storage = net_storage
        res_storage[t] = current_storage
        res_elevation[t] = storage_to_elevation_meters(current_storage)
        turbine_discharge[t] = actual_turb_q
        spill_discharge[t] = actual_spill_q

    # Calculate exact physics-based power generation
    power_mw, net_head, efficiency = calculate_power_output(turbine_discharge, res_elevation)
    water_rate = calculate_water_rate(power_mw, turbine_discharge)

    # Assemble comprehensive dataset
    df = pd.DataFrame({
        "Date": dates,
        "Water Inflow (m³/s)": inflow,
        "Rainfall (mm)": weather_df["Rainfall (mm)"].values,
        "Temp_Mean (°C)": weather_df["Temp_Mean (C)"].values,
        "Turbine Discharge (m³/s)": np.round(turbine_discharge, 2),
        "Spillway Discharge (m³/s)": np.round(spill_discharge, 2),
        "Reservoir Level (m)": np.round(res_elevation, 2),
        "Storage (MCM)": np.round(res_storage, 1),
        "Live Storage %": np.round(100.0 * (res_storage - DEAD_STORAGE_MCM) / LIVE_STORAGE_MCM, 1),
        "Net Head (m)": np.round(net_head, 2),
        "Turbine Efficiency": np.round(efficiency, 3),
        "Power Output (MW)": np.round(power_mw, 2),
        "Water Rate (m³/MWh)": np.round(water_rate, 2),
        "Daily Energy (MWh)": np.round(power_mw * 24.0, 1)
    })

    return df


if __name__ == "__main__":
    df = simulate_dam_operations()
    print("Simulated dataset shape:", df.shape)
    print(df.head())
    print("\nSummary Statistics:")
    print(df[["Water Inflow (m³/s)", "Reservoir Level (m)", "Power Output (MW)", "Water Rate (m³/MWh)"]].describe())
