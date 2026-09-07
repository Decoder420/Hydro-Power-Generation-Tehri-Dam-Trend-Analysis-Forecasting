"""Hydroelectric Dispatcher, Reservoir Rule Curve Engine, and Day-Ahead Grid Scheduler.

Models:
1. Rule Curve Compliance & Flood Buffer Alerts (CWC guidelines for Tehri Dam).
2. Day-Ahead 96-Time-Block (15-minute) Generation Schedule for NRLDC (Northern Regional Load Despatch Centre).
3. "What-If" Hydrological Scenario Simulation (Extreme precipitation storms, cloudbursts, and dry spells).
4. Deviation Settlement Mechanism (DSM) Risk Bands.
"""

from typing import Dict, Any, List
import numpy as np
import pandas as pd

from src.physics import (
    calculate_power_output,
    calculate_water_rate,
    elevation_to_storage_mcm,
    storage_to_elevation_meters,
    FRL,
    MDDL,
    MAX_DISCHARGE,
    MAX_CAPACITY_MW,
    TEHRI_SPECS
)


class TehriRuleCurveEngine:
    """Monitors reservoir status against seasonal upper and lower rule curves."""

    MONSOON_FLOOD_CUSHION = TEHRI_SPECS["rule_curves"]["monsoon_flood_cushion_level_meters"] # 825.0 m
    POST_MONSOON_TARGET = TEHRI_SPECS["rule_curves"]["post_monsoon_target_level_meters"]       # 830.0 m
    DRY_SEASON_BUFFER = TEHRI_SPECS["rule_curves"]["dry_season_buffer_level_meters"]          # 745.0 m

    @classmethod
    def evaluate_reservoir_status(cls, elevation: float, month: int) -> Dict[str, Any]:
        """Evaluate operating safety and flood buffer status."""
        status = "NORMAL"
        severity = "info"
        recommendation = "Continue planned merit-order dispatch."

        # Safety checks
        if elevation >= FRL:
            status = "CRITICAL: FRL BREACH"
            severity = "danger"
            recommendation = "Engage chute spillways immediately to prevent dam overtopping!"
        elif month in [7, 8, 9] and elevation > cls.MONSOON_FLOOD_CUSHION:
            status = "WARNING: MONSOON FLOOD BUFFER ENCROACHED"
            severity = "warning"
            recommendation = (
                f"Reservoir at {elevation:.1f} m exceeds monsoon flood cushion ({cls.MONSOON_FLOOD_CUSHION} m). "
                "Increase turbine discharge to 100% (500 m³/s) to absorb upstream storm runoff."
            )
        elif elevation <= MDDL:
            status = "CRITICAL: MDDL BREACH"
            severity = "danger"
            recommendation = "Halt power generation! Water level at or below Minimum Drawdown Level (740 m)."
        elif elevation < cls.DRY_SEASON_BUFFER:
            status = "ADVISORY: LOW RESERVOIR STORAGE"
            severity = "warning"
            recommendation = "Conserve water for essential drinking supply and high-value peak grid hours."

        storage_mcm = elevation_to_storage_mcm(elevation)
        live_storage_mcm = max(0.0, storage_mcm - TEHRI_SPECS["reservoir"]["dead_storage_mcm"])
        live_pct = min(100.0, (live_storage_mcm / TEHRI_SPECS["reservoir"]["live_storage_mcm"]) * 100.0)

        return {
            "elevation_meters": round(elevation, 2),
            "storage_mcm": round(storage_mcm, 1),
            "live_storage_pct": round(live_pct, 1),
            "status": status,
            "severity": severity,
            "recommendation": recommendation
        }


class DayAheadScheduler:
    """Generates standard 96-time-block (15-minute) dispatch schedules for Grid-India / NRLDC."""

    @staticmethod
    def generate_96_block_schedule(
        date_str: str,
        reservoir_level: float,
        daily_target_energy_mwh: float = 8500.0,
        inflow_m3_per_s: float = 120.0
    ) -> pd.DataFrame:
        """Generate 96 blocks (15-minute intervals) for a 24-hour day-ahead schedule.

        Tehri acts as a flexible peaking plant for the Northern Indian Grid:
        - Off-peak (00:00 - 06:00): Low baseload (~150 - 250 MW)
        - Morning Peak (06:00 - 10:00): High dispatch (~700 - 950 MW)
        - Solar Hours (10:00 - 17:00): Moderated dispatch (~300 - 500 MW)
        - Evening Peak (18:00 - 22:30): Maximum dispatch (~850 - 1000 MW)
        - Late Night (22:30 - 24:00): Tapering down (~250 - 350 MW)
        """
        # 96 blocks of 15 minutes each
        blocks = np.arange(1, 97)
        times = []
        base_weights = np.zeros(96)

        for b in blocks:
            minute_of_day = (b - 1) * 15
            hour = minute_of_day // 60
            minute = minute_of_day % 60
            times.append(f"{hour:02d}:{minute:02d}")

            # Define hourly dispatch profile weight
            if 0 <= hour < 6:
                base_weights[b - 1] = 0.25
            elif 6 <= hour < 10:
                base_weights[b - 1] = 0.90
            elif 10 <= hour < 17:
                base_weights[b - 1] = 0.45
            elif 17 <= hour < 22:
                base_weights[b - 1] = 0.95
            else:
                base_weights[b - 1] = 0.35

        # Normalize weights to match target energy
        # Energy per block (MWh) = MW * 0.25 hours
        avg_target_mw = daily_target_energy_mwh / 24.0
        scaled_mw = (base_weights / np.mean(base_weights)) * avg_target_mw
        scheduled_mw = np.clip(scaled_mw, 100.0, MAX_CAPACITY_MW)

        # Calculate required discharge for each block given head
        net_head = max(10.0, reservoir_level - 598.0)
        # Power MW ~ 0.82 * 9.81 * Q * H / 1000 => Q ~ (MW * 1000) / (0.82 * 9.81 * H)
        req_q = (scheduled_mw * 1000.0) / (0.88 * 9.80665 * net_head)
        req_q = np.clip(req_q, 40.0, MAX_DISCHARGE)

        # Actual physics-based power
        actual_power, _, eff = calculate_power_output(req_q, reservoir_level)
        water_rate = calculate_water_rate(actual_power, req_q)

        # DSM Deviation Risk Bands (+/- 12% under IEGC DSM regulations)
        dsm_band_high = actual_power * 1.12
        dsm_band_low = actual_power * 0.88

        df = pd.DataFrame({
            "Block": blocks,
            "Time": times,
            "Scheduled_MW": np.round(actual_power, 1),
            "DSM_Band_Low_MW": np.round(dsm_band_low, 1),
            "DSM_Band_High_MW": np.round(dsm_band_high, 1),
            "Discharge_Req_m3_s": np.round(req_q, 1),
            "Water_Rate_m3_MWh": np.round(water_rate, 1),
            "Energy_MWh": np.round(actual_power * 0.25, 2)
        })

        return df


def simulate_what_if_scenario(
    current_elevation: float,
    rainfall_scenario_mm: float,
    horizon_days: int = 7,
    daily_turbine_q: float = 350.0
) -> pd.DataFrame:
    """Simulate a 7-day storm or drought event to project reservoir trajectory and risk."""
    current_storage = elevation_to_storage_mcm(current_elevation)
    M3_S_TO_MCM_DAY = 0.0864

    # Catchment runoff: Tehri catchment 7511 km^2
    # Storm runoff distributed over days 1 to 4
    daily_storm_rain = np.array([
        rainfall_scenario_mm * 0.40,
        rainfall_scenario_mm * 0.35,
        rainfall_scenario_mm * 0.15,
        rainfall_scenario_mm * 0.10
    ] + [0.0] * (horizon_days - 4))[:horizon_days]

    records = []
    for day in range(1, horizon_days + 1):
        rain = daily_storm_rain[day - 1]
        storm_inflow = (rain * 9.5) + 90.0  # Baseflow + storm runoff
        inflow_mcm = storm_inflow * M3_S_TO_MCM_DAY

        turb_q = min(daily_turbine_q, MAX_DISCHARGE)
        turb_mcm = turb_q * M3_S_TO_MCM_DAY

        net_storage = current_storage + inflow_mcm - turb_mcm - 0.45
        max_storage = elevation_to_storage_mcm(FRL)

        if net_storage > max_storage:
            spill_mcm = net_storage - max_storage
            spill_q = spill_mcm / M3_S_TO_MCM_DAY
            net_storage = max_storage
        else:
            spill_q = 0.0

        current_storage = net_storage
        elev = storage_to_elevation_meters(current_storage)
        power_mw, net_head, eff = calculate_power_output(turb_q, elev)

        records.append({
            "Day": f"Day +{day}",
            "Rainfall_mm": round(rain, 1),
            "Inflow_m3_s": round(storm_inflow, 1),
            "Turbine_Q_m3_s": round(turb_q, 1),
            "Spillway_Q_m3_s": round(spill_q, 1),
            "Reservoir_Elevation_m": round(elev, 2),
            "Power_Output_MW": round(power_mw, 1),
            "Daily_Energy_GWh": round((power_mw * 24.0) / 1000.0, 3)
        })

    return pd.DataFrame(records)
