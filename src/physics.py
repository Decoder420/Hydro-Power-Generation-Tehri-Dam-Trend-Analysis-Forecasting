"""Physics-based Hydropower and Reservoir Hydraulic Modeling for Tehri Dam.

Implements:
1. Fundamental Hydropower equation: P = eta * rho * g * Q * H_net
2. Penstock head loss calculation: h_loss = k * Q^2
3. Francis turbine non-linear efficiency hill curve: eta(Q, H)
4. Water consumption rate: m^3 / MWh
5. Elevation <-> Storage conversion (hypsometric curve) for Tehri Reservoir
"""

import json
from pathlib import Path
from typing import Tuple, Union
import numpy as np

CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "tehri_specs.json"

with open(CONFIG_PATH, "r") as f:
    TEHRI_SPECS = json.load(f)

# Physical constants
RHO_WATER = 1000.0        # kg/m^3
GRAVITY = 9.80665         # m/s^2

# Dam Parameters
FRL = TEHRI_SPECS["reservoir"]["frl_meters"]                     # 830.0 m
MDDL = TEHRI_SPECS["reservoir"]["mddl_meters"]                   # 740.0 m
TAILRACE_LEVEL = TEHRI_SPECS["reservoir"]["tailrace_normal_level_meters"] # 598.0 m
LIVE_STORAGE_MCM = TEHRI_SPECS["reservoir"]["live_storage_mcm"] # 2615 MCM
DEAD_STORAGE_MCM = TEHRI_SPECS["reservoir"]["dead_storage_mcm"] # 925 MCM
MAX_DISCHARGE = TEHRI_SPECS["powerhouse"]["total_max_discharge_m3_per_s"] # 500 m^3/s
MAX_CAPACITY_MW = TEHRI_SPECS["powerhouse"]["total_installed_capacity_mw"] # 1000 MW
RATED_HEAD = TEHRI_SPECS["powerhouse"]["rated_head_meters"]       # 230 m
NOMINAL_EFF = TEHRI_SPECS["powerhouse"]["nominal_efficiency"]     # 0.92

# Penstock head loss coefficient (k_loss * Q^2 gives head loss in meters)
HEAD_LOSS_K = 0.000028


def calculate_net_head(
    reservoir_level: Union[float, np.ndarray],
    discharge: Union[float, np.ndarray],
    tailrace_level: float = TAILRACE_LEVEL
) -> Union[float, np.ndarray]:
    """Calculate net hydraulic head available across the turbines (meters).

    H_net = H_reservoir - H_tailrace - h_loss(Q)
    """
    gross_head = np.maximum(0.0, np.asarray(reservoir_level) - tailrace_level)
    h_loss = HEAD_LOSS_K * (np.asarray(discharge) ** 2)
    net_head = np.maximum(0.0, gross_head - h_loss)
    return float(net_head) if np.isscalar(reservoir_level) and np.isscalar(discharge) else net_head


def francis_turbine_efficiency(
    discharge: Union[float, np.ndarray],
    net_head: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate non-linear Francis turbine hill chart efficiency eta(Q, H).

    Francis turbines exhibit peak efficiency (~93-94%) near rated head and 85-95% discharge.
    Efficiency drops off at part-load and non-optimal head ratios.
    """
    q_arr = np.asarray(discharge, dtype=float)
    h_arr = np.asarray(net_head, dtype=float)

    # Normalized operating ratios
    q_ratio = np.clip(q_arr / MAX_DISCHARGE, 0.0, 1.1)
    h_ratio = np.clip(h_arr / RATED_HEAD, 0.4, 1.2)

    # 2D quadratic hill curve approximation
    # Peak at q_ratio ~ 0.90, h_ratio ~ 1.00
    q_term = -1.2 * ((q_ratio - 0.90) ** 2)
    h_term = -0.6 * ((h_ratio - 1.00) ** 2)

    # Part-load penalty for very low discharge (running below 30% unit capacity)
    part_load_penalty = np.where(q_ratio < 0.25, -0.25 * (0.25 - q_ratio), 0.0)

    eta = NOMINAL_EFF + 0.02 + q_term + h_term + part_load_penalty
    eta = np.clip(eta, 0.65, 0.94)

    # If zero discharge, zero generation
    eta = np.where(q_arr <= 0.0, 0.0, eta)

    return float(eta) if np.isscalar(discharge) and np.isscalar(net_head) else eta


def calculate_power_output(
    discharge: Union[float, np.ndarray],
    reservoir_level: Union[float, np.ndarray]
) -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray], Union[float, np.ndarray]]:
    """Calculate electrical power generation in Megawatts (MW) using hydropower physics.

    Formula:
        P (MW) = (eta * rho * g * Q * H_net) / 1,000,000

    Returns:
        power_mw: Capped at plant capacity (1000 MW)
        net_head: Net hydraulic head in meters
        efficiency: Turbine-generator efficiency
    """
    q_arr = np.asarray(discharge, dtype=float)
    h_res = np.asarray(reservoir_level, dtype=float)

    net_head = calculate_net_head(h_res, q_arr)
    efficiency = francis_turbine_efficiency(q_arr, net_head)

    # Power in Watts = eta * rho * g * Q * H
    power_watts = efficiency * RHO_WATER * GRAVITY * q_arr * net_head
    power_mw = power_watts / 1e6

    # Cap by total installed capacity
    power_mw = np.clip(power_mw, 0.0, MAX_CAPACITY_MW)

    if np.isscalar(discharge) and np.isscalar(reservoir_level):
        return float(power_mw), float(net_head), float(efficiency)
    return power_mw, net_head, efficiency


def calculate_water_rate(
    power_mw: Union[float, np.ndarray],
    discharge: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Calculate the specific water consumption rate in m^3 of water per MWh generated.

    Lower water rate means higher operational efficiency (generating more MWh with less water).
    """
    p_arr = np.asarray(power_mw, dtype=float)
    q_arr = np.asarray(discharge, dtype=float)

    # Volume per hour (m^3/h) = Q (m^3/s) * 3600
    water_rate = np.where(p_arr > 0.1, (q_arr * 3600.0) / p_arr, np.nan)
    return float(water_rate) if np.isscalar(power_mw) and np.isscalar(discharge) else water_rate


def elevation_to_storage_mcm(elevation: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Tehri Reservoir Elevation (meters MSL) to Gross Storage (Million Cubic Meters, MCM).

    Based on the Tehri hypsometric stage-storage curve:
    - At MDDL (740.0 m): Storage = 925 MCM (Dead storage)
    - At FRL (830.0 m): Storage = 3540 MCM (Gross storage)
    - Non-linear expansion of reservoir surface area with elevation.
    """
    elev = np.asarray(elevation, dtype=float)
    fraction = np.clip((elev - MDDL) / (FRL - MDDL), 0.0, 1.1)
    # Reservoir bowl shape follows power law exponent ~ 1.35
    live_storage = LIVE_STORAGE_MCM * (fraction ** 1.35)
    total_storage = DEAD_STORAGE_MCM + live_storage
    return float(total_storage) if np.isscalar(elevation) else total_storage


def storage_to_elevation_meters(storage_mcm: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """Tehri Reservoir Storage (MCM) to Water Surface Elevation (meters MSL)."""
    s_arr = np.asarray(storage_mcm, dtype=float)
    live_s = np.clip(s_arr - DEAD_STORAGE_MCM, 0.0, LIVE_STORAGE_MCM * 1.2)
    fraction = (live_s / LIVE_STORAGE_MCM) ** (1.0 / 1.35)
    elevation = MDDL + fraction * (FRL - MDDL)
    return float(elevation) if np.isscalar(storage_mcm) else elevation
