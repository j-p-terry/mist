"""
mixed_ice_transport_1p1d.py

What this does
--------------
- Radial transport is Eulerian and 1D in surface density.
- Phase changes are informed by a prescribed vertical structure T(r,z),
    shielding A_V(r,z), and carrier-dependent dust scale heights.
- Two solid carriers are included:
        1. pebbles: settled, fast radial drift, dominate mass transport
        2. small grains: vertically extended, nearly gas-coupled, dominate
            JWST-like ice absorption
- The model tracks CO, CO2, and H2O vapor plus mixed/trapped ice reservoirs:
        CO pure ice
        CO@CO2 ice
        CO@H2O ice
        CO2 pure ice
        CO2@H2O ice
        H2O ice

The "1+1D" approximation
------------------------
The evolved variables are radial surface densities Sigma_i(r,t). At every radius
we compute a vertical disk column, snow surfaces, and carrier-weighted survival
fractions. Those vertically averaged survival fractions determine the phase
source terms. Optional 2D snapshot files reconstruct where the reservoirs would
sit vertically, but the transport itself remains 1D radial.

Usage
-----
    python mixed_ice_transport_1p1d.py params_1p1d.yaml

Outputs
-------
    <output_dir>/snapshots/snapshot_XXXXXX.csv
    <output_dir>/snapshots_2d/snapshot2d_XXXXXX.npz        [optional]
    <output_dir>/diagnostics.csv
    <output_dir>/resolved_params.yaml
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple, Optional

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise ImportError("This script requires PyYAML. Install with: pip install pyyaml") from exc


# -----------------------------
# Constants, cgs
# -----------------------------
G = 6.67430e-8
KB = 1.380649e-16
M_H = 1.6735575e-24
AU = 1.495978707e13
YR = 365.25 * 24.0 * 3600.0
MSUN = 1.98847e33
MEARTH = 5.9722e27
PI = math.pi

MOLECULAR_WEIGHTS = {
    "CO": 28.0101,
    "CO2": 44.0095,
    "H2O": 18.01528,
}

CARRIERS = ("pebble", "small")

VAPOR_FIELDS = ("CO_gas", "CO2_gas", "H2O_gas")

# Reservoir keys used in vertical/snow-surface calculations.
SURVIVAL_KEYS = (
    "CO_pure",
    "CO_at_CO2",
    "CO_at_H2O",
    "CO2_pure",
    "CO2_at_H2O",
    "H2O",
)


def ref_field(carrier: str) -> str:
    return f"ref_solid_{carrier}"


def h2o_field(carrier: str) -> str:
    return f"H2O_ice_{carrier}"


def co2_pure_field(carrier: str) -> str:
    return f"CO2_pure_ice_{carrier}"


def co2_at_h2o_field(carrier: str) -> str:
    return f"CO2_at_H2O_ice_{carrier}"


def co_pure_field(carrier: str) -> str:
    return f"CO_pure_ice_{carrier}"


def co_at_co2_field(carrier: str) -> str:
    return f"CO_at_CO2_ice_{carrier}"


def co_at_h2o_field(carrier: str) -> str:
    return f"CO_at_H2O_ice_{carrier}"


def solid_fields_for(carrier: str) -> Tuple[str, ...]:
    return (
        ref_field(carrier),
        h2o_field(carrier),
        co2_pure_field(carrier),
        co2_at_h2o_field(carrier),
        co_pure_field(carrier),
        co_at_co2_field(carrier),
        co_at_h2o_field(carrier),
    )


SOLID_FIELDS = tuple(field for c in CARRIERS for field in solid_fields_for(c))
ALL_FIELDS = VAPOR_FIELDS + SOLID_FIELDS

FIELD_TO_CARRIER: Dict[str, str] = {}
FIELD_TO_SURVIVAL_KEY: Dict[str, str] = {}
for _c in CARRIERS:
    FIELD_TO_CARRIER[ref_field(_c)] = _c
    FIELD_TO_CARRIER[h2o_field(_c)] = _c
    FIELD_TO_CARRIER[co2_pure_field(_c)] = _c
    FIELD_TO_CARRIER[co2_at_h2o_field(_c)] = _c
    FIELD_TO_CARRIER[co_pure_field(_c)] = _c
    FIELD_TO_CARRIER[co_at_co2_field(_c)] = _c
    FIELD_TO_CARRIER[co_at_h2o_field(_c)] = _c

    FIELD_TO_SURVIVAL_KEY[h2o_field(_c)] = "H2O"
    FIELD_TO_SURVIVAL_KEY[co2_pure_field(_c)] = "CO2_pure"
    FIELD_TO_SURVIVAL_KEY[co2_at_h2o_field(_c)] = "CO2_at_H2O"
    FIELD_TO_SURVIVAL_KEY[co_pure_field(_c)] = "CO_pure"
    FIELD_TO_SURVIVAL_KEY[co_at_co2_field(_c)] = "CO_at_CO2"
    FIELD_TO_SURVIVAL_KEY[co_at_h2o_field(_c)] = "CO_at_H2O"


# -----------------------------
# Default parameters
# -----------------------------
DEFAULTS: Dict[str, Any] = {
    "simulation": {
        "name": "mixed_ice_1p1d",
        "output_dir": "mixed_ice_1p1d_outputs",
        "t_end_yr": 1.0e5,
        "save_interval_yr": None,
        "save_interval_fraction": 0.01,
        "overwrite": True,
        "progress_every": 100,
    },
    "grid": {
        "r_in_au": 0.5,
        "r_out_au": 300.0,
        "n_cells": 400,
        "spacing": "log",
    },
    "star": {
        "M_star_msun": 1.0,
    },
    "gas": {
        # Fixed H2/He gas mass between r_in and r_out.
        "M_gas_msun": 0.05,
        "surface_density_profile": "tapered_powerlaw",  # powerlaw or tapered_powerlaw
        "p": 1.0,
        "r_c_au": 100.0,
        "temperature": {
            # Midplane temperature T_mid(r) = max(T_floor, T0 * (r/r0)^(-q)).
            "T0_K": 150.0,
            "r0_au": 1.0,
            "q": 0.5,
            "T_floor_K": 10.0,
        },
        "alpha": 1.0e-3,
        "mu": 2.381,
        "schmidt_gas": 1.0,
        "velocity_mode": "viscous",  # viscous, zero, constant
        "constant_v_g_cm_s": 0.0,
    },
    "dust": {
        "carriers": {
            "pebble": {
                "stokes_mode": "constant",  # constant or epstein
                "St": 0.03,
                "grain_size_cm": 0.1,
                "rho_s_g_cm3": 1.5,
                "schmidt": 1.0,
                "include_diffusion": True,
                "refractory_to_gas": 4.5e-3,
            },
            "small": {
                "stokes_mode": "constant",
                "St": 1.0e-4,
                "grain_size_cm": 1.0e-4,
                "rho_s_g_cm3": 1.5,
                "schmidt": 1.0,
                "include_diffusion": True,
                "refractory_to_gas": 5.0e-4,
            },
        },
        # Sets how cold-limit volatile inventories are distributed across the
        # two solid carriers during phase equilibrium. These should sum to 1.
        # Pebbles dominate mass transport; small grains dominate vertical/JWST
        # observability.
        "volatile_carrier_fractions": {
            "pebble": 0.90,
            "small": 0.10,
        },
    },
    "vertical": {
        "enabled": True,
        "n_z": 128,
        "zmax_H": 5.0,
        # Dust scale height Hd/Hg = sqrt(alpha_z / (alpha_z + St)).
        "alpha_z": 1.0e-3,
        "temperature": {
            "enabled": True,
            # T_atm = T_atm_factor * T_mid.
            "T_atm_factor": 2.0,
            # Transition height in units of gas scale height.
            "zq_H": 2.0,
            "power": 2.0,
        },
        "shielding": {
            "enabled": True,
            # Approximate A_V per one-sided gas column [g cm^-2].
            # ISM-like value is roughly 200-230, but disk dust can differ.
            "Av_per_g_cm2": 228.0,
            "Av_crit": 1.0,
            "Av_width": 0.3,
            "apply_to_phase": True,
        },
        "snow_surface_threshold": 0.5,
    },
    "volatiles": {
        "include_vapor_diffusion": True,
        "phase_relaxation_time_yr": 1.0,
        "CO": {
            "total_mass_fraction": 2.0e-3,
            "solid_fractions": {
                "pure": 0.40,
                "at_CO2": 0.40,
                "at_H2O": 0.20,
            },
            "release_temperatures_K": {
                "pure": 25.0,
                "at_CO2": 70.0,
                "at_H2O": 130.0,
            },
            "transition_widths_K": {
                "pure": 3.0,
                "at_CO2": 6.0,
                "at_H2O": 10.0,
            },
        },
        "CO2": {
            "total_mass_fraction": 1.0e-3,
            "solid_fractions": {
                "pure": 0.75,
                "at_H2O": 0.25,
            },
            "release_temperatures_K": {
                "pure": 70.0,
                "at_H2O": 130.0,
            },
            "transition_widths_K": {
                "pure": 6.0,
                "at_H2O": 10.0,
            },
        },
        "H2O": {
            "total_mass_fraction": 5.0e-3,
            "release_temperature_K": 150.0,
            "transition_width_K": 10.0,
        },
    },
    "initial_conditions": {
        "initialize_phase_equilibrium": True,
    },
    "output": {
        "save_1d_csv": True,
        "save_2d_npz": True,
        "save_2d_every_n_snapshots": 5,
    },
    "numerics": {
        "cfl_advective": 0.4,
        "cfl_diffusive": 0.2,
        "max_timestep_yr": 50.0,
        "surface_density_floor": 0.0,
        "boundary_condition": "outflow",
        "clip_negative": True,
        "negative_tolerance": 1.0e-40,
    },
}


# -----------------------------
# Utility functions
# -----------------------------
def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_params(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        user = yaml.safe_load(f) or {}
    params = deep_update(DEFAULTS, user)
    validate_params(params)
    return params


def validate_params(p: Dict[str, Any]) -> None:
    if p["grid"]["spacing"] not in {"log", "linear"}:
        raise ValueError("grid.spacing must be 'log' or 'linear'.")
    if p["grid"]["r_in_au"] <= 0 or p["grid"]["r_out_au"] <= p["grid"]["r_in_au"]:
        raise ValueError("Require 0 < r_in_au < r_out_au.")
    if int(p["grid"]["n_cells"]) < 8:
        raise ValueError("grid.n_cells should be at least 8.")

    if p["gas"]["surface_density_profile"] not in {"powerlaw", "tapered_powerlaw"}:
        raise ValueError("gas.surface_density_profile must be 'powerlaw' or 'tapered_powerlaw'.")
    if p["gas"]["velocity_mode"] not in {"viscous", "zero", "constant"}:
        raise ValueError("gas.velocity_mode must be 'viscous', 'zero', or 'constant'.")

    for carrier in CARRIERS:
        cfg = p["dust"]["carriers"][carrier]
        if cfg["stokes_mode"] not in {"constant", "epstein"}:
            raise ValueError(f"dust.carriers.{carrier}.stokes_mode must be 'constant' or 'epstein'.")

    cf = p["dust"]["volatile_carrier_fractions"]
    carrier_sum = sum(float(cf[c]) for c in CARRIERS)
    if not np.isclose(carrier_sum, 1.0, rtol=0.0, atol=1.0e-10):
        raise ValueError(f"dust.volatile_carrier_fractions must sum to 1; got {carrier_sum:.6g}.")

    for species in ("CO", "CO2"):
        frac_sum = sum(float(x) for x in p["volatiles"][species]["solid_fractions"].values())
        if frac_sum > 1.0 + 1e-12:
            raise ValueError(f"{species} solid_fractions sum to {frac_sum:.6g} > 1.")
        if frac_sum < 1.0 - 1e-12:
            print(
                f"WARNING: {species} solid_fractions sum to {frac_sum:.6g} < 1; "
                "the cold-limit remainder can remain vapor.",
                file=sys.stderr,
            )

    if p["volatiles"]["phase_relaxation_time_yr"] <= 0:
        raise ValueError("volatiles.phase_relaxation_time_yr must be positive.")

    if bool(p["vertical"]["enabled"]):
        if int(p["vertical"]["n_z"]) < 8:
            raise ValueError("vertical.n_z should be at least 8.")
        if float(p["vertical"]["zmax_H"]) <= 0:
            raise ValueError("vertical.zmax_H must be positive.")

    if p["simulation"]["save_interval_yr"] is None:
        frac = float(p["simulation"]["save_interval_fraction"])
        if frac <= 0:
            raise ValueError("simulation.save_interval_fraction must be positive.")


def ensure_output_dir(path: Path, overwrite: bool, save_2d: bool) -> None:
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {path} exists and is non-empty. "
            "Set simulation.overwrite: true or choose a new output_dir."
        )
    (path / "snapshots").mkdir(parents=True, exist_ok=True)
    if save_2d:
        (path / "snapshots_2d").mkdir(parents=True, exist_ok=True)


def write_resolved_params(params: Dict[str, Any], output_dir: Path) -> None:
    with open(output_dir / "resolved_params.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(params, f, sort_keys=False)


def annulus_integral_sigma(area: np.ndarray, sigma: np.ndarray) -> float:
    return float(np.sum(area * sigma))


def smooth_ice_fraction(T: np.ndarray, T_release: float, width: float) -> np.ndarray:
    width = max(float(width), 1.0e-12)
    return 0.5 * (1.0 - np.tanh((T - float(T_release)) / width))


def erf_array(x: np.ndarray) -> np.ndarray:
    """Vectorized math.erf without requiring scipy."""
    return np.vectorize(math.erf, otypes=[float])(x)


def erfc_array(x: np.ndarray) -> np.ndarray:
    """Vectorized math.erfc without requiring scipy."""
    return np.vectorize(math.erfc, otypes=[float])(x)


# -----------------------------
# Grid and fixed disk
# -----------------------------
def make_grid(params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    g = params["grid"]
    n = int(g["n_cells"])
    rin = float(g["r_in_au"]) * AU
    rout = float(g["r_out_au"]) * AU

    if g["spacing"] == "log":
        r_edge = np.geomspace(rin, rout, n + 1)
        r = np.sqrt(r_edge[:-1] * r_edge[1:])
    else:
        r_edge = np.linspace(rin, rout, n + 1)
        r = 0.5 * (r_edge[:-1] + r_edge[1:])

    dr_center = np.empty_like(r)
    dr_center[1:-1] = 0.5 * (r[2:] - r[:-2])
    dr_center[0] = r[1] - r[0]
    dr_center[-1] = r[-1] - r[-2]

    area = PI * (r_edge[1:] ** 2 - r_edge[:-1] ** 2)
    return {
        "r": r,
        "r_au": r / AU,
        "r_edge": r_edge,
        "r_edge_au": r_edge / AU,
        "dr": dr_center,
        "area": area,
    }


def gas_shape(r: np.ndarray, params: Dict[str, Any]) -> np.ndarray:
    gas = params["gas"]
    p = float(gas["p"])
    rc = float(gas["r_c_au"]) * AU
    x = r / rc

    if gas["surface_density_profile"] == "powerlaw":
        shape = x ** (-p)
    elif gas["surface_density_profile"] == "tapered_powerlaw":
        exponent = max(2.0 - p, 1.0e-8)
        shape = x ** (-p) * np.exp(-(x ** exponent))
    else:
        raise ValueError("Unknown surface density profile.")

    return shape


def build_fixed_disk(grid: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r = grid["r"]
    area = grid["area"]
    gas = params["gas"]

    shape = gas_shape(r, params)
    m_target = float(gas["M_gas_msun"]) * MSUN
    sigma_norm = m_target / np.sum(area * shape)
    Sigma_g = sigma_norm * shape

    temp = gas["temperature"]
    T0 = float(temp["T0_K"])
    r0 = float(temp["r0_au"]) * AU
    q = float(temp["q"])
    T_floor = float(temp["T_floor_K"])
    T_mid = np.maximum(T_floor, T0 * (r / r0) ** (-q))

    M_star = float(params["star"]["M_star_msun"]) * MSUN
    Omega = np.sqrt(G * M_star / r**3)
    v_K = r * Omega
    c_s = np.sqrt(KB * T_mid / (float(gas["mu"]) * M_H))
    H = c_s / Omega
    nu = float(gas["alpha"]) * c_s * H

    if gas["velocity_mode"] == "viscous":
        v_g = -1.5 * nu / r
    elif gas["velocity_mode"] == "zero":
        v_g = np.zeros_like(r)
    else:
        v_g = np.full_like(r, float(gas["constant_v_g_cm_s"]))

    rho_mid = Sigma_g / (np.sqrt(2.0 * PI) * H)
    P_mid = rho_mid * c_s**2
    dlnP_dlnr = np.gradient(np.log(P_mid), np.log(r), edge_order=2)
    eta = -0.5 * (c_s / v_K) ** 2 * dlnP_dlnr
    D_g = float(gas["alpha"]) * c_s * H / float(gas["schmidt_gas"])

    return {
        "Sigma_g": Sigma_g,
        "T_mid": T_mid,
        "Omega": Omega,
        "v_K": v_K,
        "c_s": c_s,
        "H": H,
        "nu": nu,
        "v_g": v_g,
        "eta": eta,
        "D_g": D_g,
        "rho_mid": rho_mid,
        "P_mid": P_mid,
        "dlnP_dlnr": dlnP_dlnr,
    }


def build_carrier_coefficients(
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, Dict[str, np.ndarray]]:
    out: Dict[str, Dict[str, np.ndarray]] = {}
    Sigma_g = disk["Sigma_g"]

    for carrier in CARRIERS:
        cfg = params["dust"]["carriers"][carrier]
        if cfg["stokes_mode"] == "constant":
            St = np.full_like(Sigma_g, float(cfg["St"]))
        else:
            a = float(cfg["grain_size_cm"])
            rho_s = float(cfg["rho_s_g_cm3"])
            St = 0.5 * PI * rho_s * a / np.maximum(Sigma_g, 1.0e-300)

        v_g = disk["v_g"]
        eta = disk["eta"]
        v_K = disk["v_K"]
        v = (v_g - 2.0 * eta * v_K * St) / (1.0 + St**2)

        D = disk["D_g"] / (1.0 + St**2) / float(cfg["schmidt"])

        alpha_z = float(params["vertical"]["alpha_z"])
        H_ratio = np.sqrt(alpha_z / np.maximum(alpha_z + St, 1.0e-300))
        H_ratio = np.clip(H_ratio, 1.0e-4, 1.0)

        out[carrier] = {
            "St": St,
            "v": v,
            "D": D,
            "H_ratio": H_ratio,
        }

    return out

# -----------------------------
# Trapping capacity enforcement
# -----------------------------

def trapping_capacity_config(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return trapping-capacity config with safe defaults.
    """
    cfg = params.get("volatiles", {}).get("trapping_capacity", {})
    if isinstance(cfg, bool):
        return {"enabled": cfg}
    if cfg is None:
        return {"enabled": False}
    return cfg


def _max_guest_surface_density_from_mol_ratio(
    host_sigma: np.ndarray,
    guest_species: str,
    host_species: str,
    max_guest_per_host_mol: float,
) -> np.ndarray:
    """
    Convert a molecular guest/host capacity into a guest mass surface-density cap.

    If max_guest_per_host_mol = N_guest / N_host, then

        Sigma_guest_max =
            max_guest_per_host_mol * (mu_guest / mu_host) * Sigma_host.
    """
    mu_guest = MOLECULAR_WEIGHTS[guest_species]
    mu_host = MOLECULAR_WEIGHTS[host_species]

    return float(max_guest_per_host_mol) * (mu_guest / mu_host) * host_sigma


def _cap_guest_reservoir_to_host(
    targets: Dict[str, np.ndarray],
    guest_field: str,
    gas_field: str,
    host_sigma: np.ndarray,
    guest_species: str,
    host_species: str,
    max_guest_per_host_mol: Optional[float],
) -> np.ndarray:
    """
    Cap one guest ice reservoir by available host matrix.

    Excess guest mass is moved back to the corresponding gas reservoir.
    Returns the excess surface density.
    """
    if max_guest_per_host_mol is None:
        return np.zeros_like(host_sigma)

    max_guest_per_host_mol = float(max_guest_per_host_mol)

    if max_guest_per_host_mol < 0.0:
        raise ValueError(
            f"max_guest_per_host_mol must be >= 0 for {guest_field}; "
            f"got {max_guest_per_host_mol}"
        )

    guest_before = np.maximum(targets[guest_field], 0.0)

    guest_cap = _max_guest_surface_density_from_mol_ratio(
        host_sigma=host_sigma,
        guest_species=guest_species,
        host_species=host_species,
        max_guest_per_host_mol=max_guest_per_host_mol,
    )

    guest_after = np.minimum(guest_before, guest_cap)
    excess = np.maximum(guest_before - guest_after, 0.0)

    targets[guest_field] = guest_after
    targets[gas_field] = np.maximum(targets[gas_field], 0.0) + excess

    return excess


def apply_trapping_capacity_limits(
    targets: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Enforce finite host-matrix capacity for mixed ice reservoirs.

    This should be called at the end of phase_targets(...), after the ordinary
    equilibrium targets have already been computed.

    Capacity limits are applied per carrier, so pebble CO@H2O can only use
    pebble H2O ice as host, and small-grain CO@H2O can only use small-grain
    H2O ice as host.

    Supports two excess-destination modes:

      excess_destination: gas
          Excess guest material is returned directly to the corresponding gas.

      excess_destination: next_available_reservoir
          Excess guest material is first moved into another locally available
          solid reservoir of the same molecule, subject to its own capacity
          limits, and only the leftover is returned to gas.

          For CO excess, the fallback order is:
              CO@H2O -> CO@CO2 -> pure CO -> CO gas

          For CO2 excess, the fallback order is:
              pure CO2 -> CO2 gas

          The source reservoir is excluded from its own fallback list.
    """

    cfg = trapping_capacity_config(params)

    if not bool(cfg.get("enabled", False)):
        return targets

    excess_destination = str(cfg.get("excess_destination", "gas"))
    allowed_destinations = {"gas", "next_available_reservoir"}
    if excess_destination not in allowed_destinations:
        raise ValueError(
            "Unknown volatiles.trapping_capacity.excess_destination="
            f"{excess_destination!r}. Use one of {sorted(allowed_destinations)}."
        )

    host_floor = float(cfg.get("host_floor", 1.0e-300))

    # Used only to decide whether an uncapped pure reservoir is meaningfully
    # available. This prevents a tiny numerical floor from making pure ice
    # available everywhere.
    availability_relative_floor = float(cfg.get("availability_relative_floor", 1.0e-12))
    availability_abs_floor = float(cfg.get("availability_abs_floor", host_floor))

    co_at_co2_cfg = cfg.get("CO_at_CO2", {})
    co_at_h2o_cfg = cfg.get("CO_at_H2O", {})
    co2_at_h2o_cfg = cfg.get("CO2_at_H2O", {})
    h2o_total_cfg = cfg.get("H2O_total_guest_capacity", {})

    co_at_co2_max = co_at_co2_cfg.get("max_guest_per_host_mol", None)
    co_at_h2o_max = co_at_h2o_cfg.get("max_guest_per_host_mol", None)
    co2_at_h2o_max = co2_at_h2o_cfg.get("max_guest_per_host_mol", None)

    co2_host_mode = co_at_co2_cfg.get("host_mode", "total_CO2_ice")
    total_h2o_guest_max = h2o_total_cfg.get("max_total_guest_per_host_mol", None)

    def _positive(field: str) -> np.ndarray:
        return np.maximum(targets[field], 0.0)

    def _meaningfully_present(arr: np.ndarray) -> np.ndarray:
        arr = np.maximum(arr, 0.0)
        if not np.any(arr > 0.0):
            return np.zeros_like(arr, dtype=bool)
        threshold = max(availability_abs_floor, availability_relative_floor * float(np.nanmax(arr)))
        return arr > threshold

    def _gas_field_for_species(species: str) -> str:
        if species == "CO":
            return "CO_gas"
        if species == "CO2":
            return "CO2_gas"
        if species == "H2O":
            return "H2O_gas"
        raise ValueError(f"Unknown species {species!r}.")

    def _co2_host_for_carrier(carrier: str) -> np.ndarray:
        if co2_host_mode == "pure_CO2_ice":
            return _positive(co2_pure_field(carrier))

        if co2_host_mode == "total_CO2_ice":
            return np.maximum(
                _positive(co2_pure_field(carrier))
                + _positive(co2_at_h2o_field(carrier)),
                0.0,
            )

        raise ValueError(
            "Unknown volatiles.trapping_capacity.CO_at_CO2.host_mode="
            f"{co2_host_mode!r}. Use 'pure_CO2_ice' or 'total_CO2_ice'."
        )

    def _capacity_mass_from_host(
        host_sigma: np.ndarray,
        guest_species: str,
        host_species: str,
        max_guest_per_host_mol: Optional[float],
    ) -> np.ndarray:
        """
        Return the maximum guest mass surface density allowed by the host.
        """
        if max_guest_per_host_mol is None:
            # No finite cap. Caller still decides whether host is available.
            return np.full_like(host_sigma, np.inf, dtype=float)

        n_host = np.maximum(host_sigma, 0.0) / MOLECULAR_WEIGHTS[host_species]
        n_guest_max = float(max_guest_per_host_mol) * n_host
        return n_guest_max * MOLECULAR_WEIGHTS[guest_species]

    def _remaining_capacity_for_field(
        field: str,
        carrier: str,
    ) -> np.ndarray:
        """
        Remaining mass capacity of a possible destination field.

        Pure ice fields are treated as uncapped but only available where the
        pure target reservoir is already meaningfully present.
        """
        current = _positive(field)

        # CO hosted in H2O ice.
        if field == co_at_h2o_field(carrier):
            h2o_host = _positive(h2o_field(carrier))
            host_available = _meaningfully_present(h2o_host)

            cap = _capacity_mass_from_host(
                host_sigma=h2o_host,
                guest_species="CO",
                host_species="H2O",
                max_guest_per_host_mol=co_at_h2o_max,
            )
            remaining = np.maximum(cap - current, 0.0)
            return np.where(host_available, remaining, 0.0)

        # CO hosted in CO2 ice.
        if field == co_at_co2_field(carrier):
            co2_host = _co2_host_for_carrier(carrier)
            host_available = _meaningfully_present(co2_host)

            cap = _capacity_mass_from_host(
                host_sigma=co2_host,
                guest_species="CO",
                host_species="CO2",
                max_guest_per_host_mol=co_at_co2_max,
            )
            remaining = np.maximum(cap - current, 0.0)
            return np.where(host_available, remaining, 0.0)

        # CO2 hosted in H2O ice.
        if field == co2_at_h2o_field(carrier):
            h2o_host = _positive(h2o_field(carrier))
            host_available = _meaningfully_present(h2o_host)

            cap = _capacity_mass_from_host(
                host_sigma=h2o_host,
                guest_species="CO2",
                host_species="H2O",
                max_guest_per_host_mol=co2_at_h2o_max,
            )
            remaining = np.maximum(cap - current, 0.0)
            return np.where(host_available, remaining, 0.0)

        # Pure CO ice: uncapped, but only where pure CO ice is already
        # meaningfully present in the equilibrium target.
        if field == co_pure_field(carrier):
            available = _meaningfully_present(current)
            return np.where(available, np.inf, 0.0)

        # Pure CO2 ice: uncapped, but only where pure CO2 ice is already
        # meaningfully present in the equilibrium target.
        if field == co2_pure_field(carrier):
            available = _meaningfully_present(current)
            return np.where(available, np.inf, 0.0)

        raise ValueError(f"Do not know how to compute capacity for field {field!r}.")

    def _add_to_field_with_capacity(
        field: str,
        carrier: str,
        remaining_excess: np.ndarray,
    ) -> np.ndarray:
        """
        Add as much of remaining_excess as possible to field, respecting
        the field's current remaining capacity. Return leftover excess.
        """
        remaining_excess = np.maximum(remaining_excess, 0.0)
        if not np.any(remaining_excess > 0.0):
            return remaining_excess

        capacity_left = _remaining_capacity_for_field(field, carrier)
        add = np.minimum(remaining_excess, capacity_left)

        finite_add = np.where(np.isfinite(add), add, remaining_excess)
        targets[field] = _positive(field) + finite_add

        return np.maximum(remaining_excess - finite_add, 0.0)

    def _route_excess(
        *,
        species: str,
        carrier: str,
        excess: np.ndarray,
        source_field: str,
    ) -> None:
        """
        Route excess material either to gas or to the next available solid
        reservoir of the same species.
        """
        excess = np.maximum(excess, 0.0)
        if not np.any(excess > 0.0):
            return

        gas_field = _gas_field_for_species(species)

        if excess_destination == "gas":
            targets[gas_field] = _positive(gas_field) + excess
            return

        remaining = excess.copy()

        if species == "CO":
            # Stronger/more refractory reservoirs first, then pure CO,
            # but never send material back into its source reservoir.
            candidates = [
                co_at_h2o_field(carrier),
                co_at_co2_field(carrier),
                co_pure_field(carrier),
            ]

        elif species == "CO2":
            # If CO2 cannot be stored as CO2@H2O, the only same-species
            # solid fallback in this model is pure CO2 ice.
            candidates = [
                co2_pure_field(carrier),
            ]

        else:
            candidates = []

        for dest_field in candidates:
            if dest_field == source_field:
                continue
            remaining = _add_to_field_with_capacity(dest_field, carrier, remaining)
            if not np.any(remaining > 0.0):
                break

        # Whatever cannot be stored as solid goes to gas.
        if np.any(remaining > 0.0):
            targets[gas_field] = _positive(gas_field) + remaining

    def _cap_guest_reservoir_to_host_local(
        *,
        guest_field: str,
        carrier: str,
        host_sigma: np.ndarray,
        guest_species: str,
        host_species: str,
        max_guest_per_host_mol: Optional[float],
    ) -> None:
        """
        Cap one guest reservoir against a host reservoir and route the excess.
        """
        if max_guest_per_host_mol is None:
            return

        guest = _positive(guest_field)

        max_guest_mass = _capacity_mass_from_host(
            host_sigma=host_sigma,
            guest_species=guest_species,
            host_species=host_species,
            max_guest_per_host_mol=max_guest_per_host_mol,
        )

        guest_new = np.minimum(guest, max_guest_mass)
        excess = np.maximum(guest - guest_new, 0.0)

        targets[guest_field] = guest_new

        _route_excess(
            species=guest_species,
            carrier=carrier,
            excess=excess,
            source_field=guest_field,
        )

    for carrier in CARRIERS:
        # ------------------------------------------------------------
        # Host reservoirs for this carrier
        # ------------------------------------------------------------
        h2o_host = _positive(h2o_field(carrier))

        # ------------------------------------------------------------
        # First cap CO2@H2O, since this may contribute to the available
        # CO2-bearing matrix if host_mode == total_CO2_ice.
        # ------------------------------------------------------------
        _cap_guest_reservoir_to_host_local(
            guest_field=co2_at_h2o_field(carrier),
            carrier=carrier,
            host_sigma=h2o_host,
            guest_species="CO2",
            host_species="H2O",
            max_guest_per_host_mol=co2_at_h2o_max,
        )

        co2_host = _co2_host_for_carrier(carrier)

        # ------------------------------------------------------------
        # Cap CO trapped in CO2-associated ice.
        # ------------------------------------------------------------
        _cap_guest_reservoir_to_host_local(
            guest_field=co_at_co2_field(carrier),
            carrier=carrier,
            host_sigma=co2_host,
            guest_species="CO",
            host_species="CO2",
            max_guest_per_host_mol=co_at_co2_max,
        )

        # ------------------------------------------------------------
        # Cap CO trapped in H2O-associated ice.
        # ------------------------------------------------------------
        _cap_guest_reservoir_to_host_local(
            guest_field=co_at_h2o_field(carrier),
            carrier=carrier,
            host_sigma=h2o_host,
            guest_species="CO",
            host_species="H2O",
            max_guest_per_host_mol=co_at_h2o_max,
        )

        # ------------------------------------------------------------
        # Optional total guest cap for H2O ice:
        # (N_CO@H2O + N_CO2@H2O) / N_H2O <= max_total.
        # ------------------------------------------------------------
        if total_h2o_guest_max is not None:
            max_total = float(total_h2o_guest_max)

            co_h2o_field = co_at_h2o_field(carrier)
            co2_h2o_field = co2_at_h2o_field(carrier)

            co_guest = _positive(co_h2o_field)
            co2_guest = _positive(co2_h2o_field)

            n_co_guest = co_guest / MOLECULAR_WEIGHTS["CO"]
            n_co2_guest = co2_guest / MOLECULAR_WEIGHTS["CO2"]
            n_h2o_host = np.maximum(h2o_host, host_floor) / MOLECULAR_WEIGHTS["H2O"]

            n_guest_total = n_co_guest + n_co2_guest
            n_guest_cap = max_total * n_h2o_host

            over = n_guest_total > n_guest_cap

            if np.any(over):
                scale = np.ones_like(n_guest_total)
                scale[over] = n_guest_cap[over] / np.maximum(n_guest_total[over], 1.0e-300)

                co_guest_new = co_guest * scale
                co2_guest_new = co2_guest * scale

                co_excess = np.maximum(co_guest - co_guest_new, 0.0)
                co2_excess = np.maximum(co2_guest - co2_guest_new, 0.0)

                targets[co_h2o_field] = co_guest_new
                targets[co2_h2o_field] = co2_guest_new

                _route_excess(
                    species="CO",
                    carrier=carrier,
                    excess=co_excess,
                    source_field=co_h2o_field,
                )
                _route_excess(
                    species="CO2",
                    carrier=carrier,
                    excess=co2_excess,
                    source_field=co2_h2o_field,
                )

    # Final safety pass: the routing above should respect capacities, but this
    # removes small negative roundoff and keeps the state non-negative.
    for name in ALL_FIELDS:
        targets[name] = np.maximum(targets[name], 0.0)

    return targets
# -----------------------------
# Backreaction
# -----------------------------
def get_backreaction_enabled(params):
    br = params.get("dust", {}).get("backreaction", False)
    if isinstance(br, dict):
        return bool(br.get("enabled", False))
    return bool(br)


def carrier_total_solid_sigma(state: Dict[str, np.ndarray], carrier: str) -> np.ndarray:
    """
    Total solid surface density carried by one dust population.

    Includes refractory solids plus all volatile ices.
    Does not include vapor reservoirs.
    """
    return (
        state[ref_field(carrier)]
        + state[h2o_field(carrier)]
        + state[co2_pure_field(carrier)]
        + state[co2_at_h2o_field(carrier)]
        + state[co_pure_field(carrier)]
        + state[co_at_co2_field(carrier)]
        + state[co_at_h2o_field(carrier)]
    )


def apply_dust_backreaction_to_velocities(
    state: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    params: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], Dict[str, Dict[str, np.ndarray]], Dict[str, np.ndarray]]:
    """
    Apply multi-carrier dust backreaction to gas and dust velocities.

    This implements the same X, Y, A, B structure as your Garate/Drazkowska
    helper functions, but vectorized over radius and adapted to surface-density
    1+1D quantities.

    Assumptions:
      - backreaction is local in radius;
      - dust volume density ratios are estimated from Sigma_d/Sigma_g times H_g/H_d;
      - vapors are part of the gas reservoir and do not directly contribute to dust feedback;
      - the pressure-support speed is v_P = eta * v_K;
      - disk["v_g_visc"] is used if available, otherwise disk["v_g"] is treated
        as the no-backreaction viscous gas radial velocity.
    """

    br_cfg = params.get("dust", {}).get("backreaction", {})
    enabled = bool(br_cfg.get("enabled", False))

    if not enabled:
        return disk, carrier_coeff, {}

    include_small = bool(br_cfg.get("include_small", False))
    epsilon_mode = br_cfg.get("epsilon_mode", "midplane")
    epsilon_cap = float(br_cfg.get("epsilon_cap", 10.0))
    sigma_floor = float(br_cfg.get("sigma_floor", 1.0e-300))

    Sigma_g = np.maximum(disk["Sigma_g"], sigma_floor)

    # Use the no-backreaction viscous gas velocity as the base velocity.
    # Important: do not use a previously backreacted disk["v_g"] as v_visc
    # unless you intentionally want to compound the effect.
    v_visc = disk.get("v_g_visc", disk["v_g"])

    # Pressure-support speed. In your no-backreaction dust velocity,
    # v_d = (v_g - 2 eta v_K St)/(1+St^2), so v_P = eta v_K.
    v_P = disk["eta"] * disk["v_K"]

    carriers_for_feedback = ["pebble"]
    if include_small:
        carriers_for_feedback.append("small")

    # Compute X and Y.
    X = np.zeros_like(Sigma_g)
    Y = np.zeros_like(Sigma_g)

    epsilon_by_carrier: Dict[str, np.ndarray] = {}

    for carrier in carriers_for_feedback:
        Sigma_d = carrier_total_solid_sigma(state, carrier)
        St = carrier_coeff[carrier]["St"]

        if epsilon_mode == "midplane":
            # rho_d/rho_g ~ (Sigma_d/Sigma_g) * (H_g/H_d)
            H_ratio = np.maximum(carrier_coeff[carrier]["H_ratio"], 1.0e-12)
            epsilon = (Sigma_d / Sigma_g) / H_ratio
        elif epsilon_mode == "column":
            epsilon = Sigma_d / Sigma_g
        else:
            raise ValueError(
                f"Unknown dust.backreaction.epsilon_mode={epsilon_mode!r}. "
                "Use 'midplane' or 'column'."
            )

        epsilon = np.clip(epsilon, 0.0, epsilon_cap)
        epsilon_by_carrier[carrier] = epsilon

        X += epsilon / (1.0 + St**2)
        Y += epsilon * St / (1.0 + St**2)

    denom = (1.0 + X) ** 2 + Y**2
    A = (1.0 + X) / np.maximum(denom, 1.0e-300)
    B = Y / np.maximum(denom, 1.0e-300)

    # Backreacted gas velocities.
    v_g_r = A * v_visc + 2.0 * B * v_P
    v_g_phi = 0.5 * B * v_visc - A * v_P

    disk["v_g_no_backreaction"] = v_visc.copy()
    disk["v_g"] = v_g_r
    disk["v_g_phi_subkep"] = v_g_phi
    disk["backreaction_X"] = X
    disk["backreaction_Y"] = Y
    disk["backreaction_A"] = A
    disk["backreaction_B"] = B

    # Backreacted dust radial velocities for all carriers.
    for carrier in CARRIERS:
        St = carrier_coeff[carrier]["St"]
        carrier_coeff[carrier]["v_no_backreaction"] = carrier_coeff[carrier]["v"].copy()
        carrier_coeff[carrier]["v"] = (
            v_g_r / (1.0 + St**2)
            + 2.0 * St * v_g_phi / (1.0 + St**2)
        )

    diagnostics: Dict[str, np.ndarray] = {
        "X": X,
        "Y": Y,
        "A": A,
        "B": B,
        "v_g_r": v_g_r,
        "v_g_phi": v_g_phi,
    }

    for carrier, eps in epsilon_by_carrier.items():
        diagnostics[f"epsilon_{carrier}"] = eps

    return disk, carrier_coeff, diagnostics


# -----------------------------
# Vertical structure / snow surfaces
# -----------------------------
def gaussian_bin_weights(y_edges: np.ndarray, H_ratio: np.ndarray) -> np.ndarray:
    """Two-sided normalized Gaussian mass fractions in z>=0 bins.

    The bins are in y=z/H_g. H_ratio is H_carrier/H_g with shape (n_r,).
    Returned weights have shape (n_r, n_z) and sum to 1 over z bins.
    """
    hr = np.asarray(H_ratio, dtype=float)[:, None]
    a = y_edges[None, :-1] / (np.sqrt(2.0) * np.maximum(hr, 1.0e-12))
    b = y_edges[None, 1:] / (np.sqrt(2.0) * np.maximum(hr, 1.0e-12))
    w = erf_array(b) - erf_array(a)
    row_sum = np.sum(w, axis=1)
    w = w / np.maximum(row_sum[:, None], 1.0e-300)
    return w


def make_vertical_structure(
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    params: Dict[str, Any],
) -> Dict[str, Any]:
    if not bool(params["vertical"]["enabled"]):
        raise ValueError("This 1+1D script expects vertical.enabled: true.")

    nz = int(params["vertical"]["n_z"])
    zmax_H = float(params["vertical"]["zmax_H"])
    y_edges = np.linspace(0.0, zmax_H, nz + 1)
    y = 0.5 * (y_edges[:-1] + y_edges[1:])

    nr = len(grid["r"])
    H = disk["H"]
    r = grid["r"]

    z_au = (H[:, None] * y[None, :]) / AU
    z_over_r = (H[:, None] * y[None, :]) / r[:, None]
    z_over_H = np.broadcast_to(y[None, :], (nr, nz)).copy()

    # Vertical temperature field.
    T_mid = disk["T_mid"]
    vtemp = params["vertical"]["temperature"]
    if bool(vtemp["enabled"]):
        T_atm = float(vtemp["T_atm_factor"]) * T_mid
        zq_H = max(float(vtemp["zq_H"]), 1.0e-12)
        power = float(vtemp["power"])
        fz = 1.0 - np.exp(-((y[None, :] / zq_H) ** power))
        T_rz = T_mid[:, None] + (T_atm[:, None] - T_mid[:, None]) * fz
    else:
        T_rz = np.broadcast_to(T_mid[:, None], (nr, nz)).copy()

    # Approximate shielding from one-sided gas column above height z.
    sh = params["vertical"]["shielding"]
    if bool(sh["enabled"]):
        # One-sided gas column above z for a Gaussian gas column.
        # Sigma_above(z) = 0.5 * Sigma_g * erfc(z/(sqrt(2) H)).
        sigma_above = 0.5 * disk["Sigma_g"][:, None] * erfc_array(y[None, :] / np.sqrt(2.0))
        Av = float(sh["Av_per_g_cm2"]) * sigma_above
        if bool(sh["apply_to_phase"]):
            Av_crit = float(sh["Av_crit"])
            Av_width = max(float(sh["Av_width"]), 1.0e-12)
            shield = 0.5 * (1.0 + np.tanh((Av - Av_crit) / Av_width))
        else:
            shield = np.ones_like(T_rz)
    else:
        sigma_above = np.zeros_like(T_rz)
        Av = np.full_like(T_rz, np.inf)
        shield = np.ones_like(T_rz)

    # Carrier vertical weights.
    weights: Dict[str, np.ndarray] = {
        "gas": gaussian_bin_weights(y_edges, np.ones(nr)),
    }
    for carrier in CARRIERS:
        weights[carrier] = gaussian_bin_weights(y_edges, carrier_coeff[carrier]["H_ratio"])

    # Survival maps.
    vol = params["volatiles"]
    survival: Dict[str, np.ndarray] = {}

    survival["CO_pure"] = smooth_ice_fraction(
        T_rz,
        vol["CO"]["release_temperatures_K"]["pure"],
        vol["CO"]["transition_widths_K"]["pure"],
    ) * shield
    survival["CO_at_CO2"] = smooth_ice_fraction(
        T_rz,
        vol["CO"]["release_temperatures_K"]["at_CO2"],
        vol["CO"]["transition_widths_K"]["at_CO2"],
    ) * shield
    survival["CO_at_H2O"] = smooth_ice_fraction(
        T_rz,
        vol["CO"]["release_temperatures_K"]["at_H2O"],
        vol["CO"]["transition_widths_K"]["at_H2O"],
    ) * shield
    survival["CO2_pure"] = smooth_ice_fraction(
        T_rz,
        vol["CO2"]["release_temperatures_K"]["pure"],
        vol["CO2"]["transition_widths_K"]["pure"],
    ) * shield
    survival["CO2_at_H2O"] = smooth_ice_fraction(
        T_rz,
        vol["CO2"]["release_temperatures_K"]["at_H2O"],
        vol["CO2"]["transition_widths_K"]["at_H2O"],
    ) * shield
    survival["H2O"] = smooth_ice_fraction(
        T_rz,
        vol["H2O"]["release_temperature_K"],
        vol["H2O"]["transition_width_K"],
    ) * shield

    mean_survival: Dict[str, Dict[str, np.ndarray]] = {c: {} for c in CARRIERS}
    for carrier in CARRIERS:
        w = weights[carrier]
        for key in SURVIVAL_KEYS:
            mean_survival[carrier][key] = np.sum(w * survival[key], axis=1)

    threshold = float(params["vertical"]["snow_surface_threshold"])
    snow_surfaces: Dict[str, np.ndarray] = {
        key: find_snow_surface_z_over_r(survival[key], z_over_r, threshold=threshold)
        for key in SURVIVAL_KEYS
    }

    return {
        "y_edges": y_edges,
        "y": y,
        "z_au": z_au,
        "z_over_H": z_over_H,
        "z_over_r": z_over_r,
        "T_rz": T_rz,
        "sigma_above": sigma_above,
        "Av": Av,
        "shield": shield,
        "weights": weights,
        "survival": survival,
        "mean_survival": mean_survival,
        "snow_surfaces_z_over_r": snow_surfaces,
    }


def find_snow_surface_z_over_r(
    survival_rz: np.ndarray,
    z_over_r: np.ndarray,
    threshold: float = 0.5,
) -> np.ndarray:
    """Find z/r where survival falls below threshold.

    Assumes z increases along axis=1. If midplane survival is already below
    threshold, returns 0. If survival remains above threshold to zmax, returns
    the top grid value.
    """
    nr, nz = survival_rz.shape
    out = np.zeros(nr, dtype=float)

    for i in range(nr):
        s = survival_rz[i]
        z = z_over_r[i]

        if s[0] < threshold:
            out[i] = 0.0
            continue
        below = np.where(s < threshold)[0]
        if below.size == 0:
            out[i] = z[-1]
            continue

        j = int(below[0])
        if j == 0:
            out[i] = z[0]
            continue

        s0, s1 = s[j - 1], s[j]
        z0, z1 = z[j - 1], z[j]
        if abs(s1 - s0) < 1.0e-30:
            out[i] = z1
        else:
            frac = (threshold - s0) / (s1 - s0)
            out[i] = z0 + frac * (z1 - z0)

    return out

def release_radii(r, r_edge, dM_release):
    """
    r: cell centers [au]
    r_edge: cell edges [au]
    dM_release: released CO mass per radial bin, e.g. g or Mearth
    """
    dM = np.asarray(dM_release, dtype=float)
    total = dM.sum()

    if total <= 0:
        return {
            "R_peak": np.nan,
            "R10": np.nan,
            "R50": np.nan,
            "R90": np.nan,
            "total_release": 0.0,
        }

    dlnr = np.log(r_edge[1:] / r_edge[:-1])
    dM_dlnr = dM / dlnr

    R_peak = r[np.argmax(dM_dlnr)]

    cum = np.cumsum(dM)
    R10 = np.interp(0.1 * total, cum, r)
    R50 = np.interp(0.5 * total, cum, r)
    R90 = np.interp(0.9 * total, cum, r)

    return {
        "R_peak": R_peak,
        "R10": R10,
        "R50": R50,
        "R90": R90,
        "total_release": total,
    }
    
def compute_co_release(state_before, state_after, dt, grid):
    area = grid["area"]
    r_edge = grid["r_edge_au"]
    dlnr = np.log(r_edge[1:] / r_edge[:-1])

    channels = {
        "CO_pure": [
            "CO_pure_ice_pebble",
            "CO_pure_ice_small",
        ],
        "CO_at_CO2": [
            "CO_at_CO2_ice_pebble",
            "CO_at_CO2_ice_small",
        ],
        "CO_at_H2O": [
            "CO_at_H2O_ice_pebble",
            "CO_at_H2O_ice_small",
        ],
    }

    out = {}

    for channel, fields in channels.items():
        dSigma_release = np.zeros_like(grid["r"])

        for field in fields:
            before = state_before[field]
            after = state_after[field]

            # Gross release from this reservoir during phase update.
            dSigma_release += np.maximum(before - after, 0.0)

        dM_release = area * dSigma_release
        Mdot_release = dM_release / dt

        out[f"dSigma_{channel}"] = dSigma_release
        out[f"dM_{channel}"] = dM_release
        out[f"Mdot_{channel}"] = Mdot_release
        out[f"Mdot_dlnr_{channel}"] = Mdot_release / dlnr
    
    dSigma_CO_gas_net = state_after["CO_gas"] - state_before["CO_gas"]
    dM_CO_gas = area * dSigma_CO_gas_net
    Mdot_CO_gas = dM_CO_gas / dt
    out["dSigma_CO_gas"] = dSigma_CO_gas_net
    out["dM_CO_gas"] = dM_CO_gas
    out["Mdot_CO_gas"] = Mdot_CO_gas
    out["Mdot_dlnr_CO_gas"] = Mdot_CO_gas / dlnr

    return out


# -----------------------------
# Initialization and phase targets
# -----------------------------
def initialize_state(
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    n = len(grid["r"])
    state = {name: np.zeros(n, dtype=np.float64) for name in ALL_FIELDS}
    Sigma_g = disk["Sigma_g"]

    for carrier in CARRIERS:
        state[ref_field(carrier)] = (
            float(params["dust"]["carriers"][carrier]["refractory_to_gas"]) * Sigma_g
        )

    CO_total = float(params["volatiles"]["CO"]["total_mass_fraction"]) * Sigma_g
    CO2_total = float(params["volatiles"]["CO2"]["total_mass_fraction"]) * Sigma_g
    H2O_total = float(params["volatiles"]["H2O"]["total_mass_fraction"]) * Sigma_g

    # Cold-limit initialization.
    cf = params["dust"]["volatile_carrier_fractions"]
    CO_fr = params["volatiles"]["CO"]["solid_fractions"]
    CO2_fr = params["volatiles"]["CO2"]["solid_fractions"]

    for carrier in CARRIERS:
        fc = float(cf[carrier])
        state[co_pure_field(carrier)] = fc * float(CO_fr["pure"]) * CO_total
        state[co_at_co2_field(carrier)] = fc * float(CO_fr["at_CO2"]) * CO_total
        state[co_at_h2o_field(carrier)] = fc * float(CO_fr["at_H2O"]) * CO_total

        state[co2_pure_field(carrier)] = fc * float(CO2_fr["pure"]) * CO2_total
        state[co2_at_h2o_field(carrier)] = fc * float(CO2_fr["at_H2O"]) * CO2_total

        state[h2o_field(carrier)] = fc * H2O_total

    state["CO_gas"] = CO_total - sum(
        state[co_pure_field(c)] + state[co_at_co2_field(c)] + state[co_at_h2o_field(c)]
        for c in CARRIERS
    )
    state["CO2_gas"] = CO2_total - sum(
        state[co2_pure_field(c)] + state[co2_at_h2o_field(c)] for c in CARRIERS
    )
    state["H2O_gas"] = H2O_total - sum(state[h2o_field(c)] for c in CARRIERS)

    for name in VAPOR_FIELDS:
        state[name] = np.maximum(state[name], 0.0)

    if bool(params["initial_conditions"]["initialize_phase_equilibrium"]):
        state = set_phase_equilibrium(state, vertical, params)

    return state


def total_CO(state: Dict[str, np.ndarray]) -> np.ndarray:
    return state["CO_gas"] + sum(
        state[co_pure_field(c)] + state[co_at_co2_field(c)] + state[co_at_h2o_field(c)]
        for c in CARRIERS
    )


def total_CO2(state: Dict[str, np.ndarray]) -> np.ndarray:
    return state["CO2_gas"] + sum(
        state[co2_pure_field(c)] + state[co2_at_h2o_field(c)] for c in CARRIERS
    )


def total_H2O(state: Dict[str, np.ndarray]) -> np.ndarray:
    return state["H2O_gas"] + sum(state[h2o_field(c)] for c in CARRIERS)


def phase_targets(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    targets = {name: np.zeros_like(next(iter(state.values()))) for name in ALL_FIELDS}

    # Ref solids do not phase change; target is current value.
    for carrier in CARRIERS:
        targets[ref_field(carrier)] = state[ref_field(carrier)].copy()

    cf = params["dust"]["volatile_carrier_fractions"]
    ms = vertical["mean_survival"]

    # CO targets.
    COT = total_CO(state)
    fr = params["volatiles"]["CO"]["solid_fractions"]
    solid_sum = np.zeros_like(COT)
    for carrier in CARRIERS:
        fc = float(cf[carrier])
        targets[co_pure_field(carrier)] = fc * float(fr["pure"]) * ms[carrier]["CO_pure"] * COT
        targets[co_at_co2_field(carrier)] = fc * float(fr["at_CO2"]) * ms[carrier]["CO_at_CO2"] * COT
        targets[co_at_h2o_field(carrier)] = fc * float(fr["at_H2O"]) * ms[carrier]["CO_at_H2O"] * COT
        solid_sum += (
            targets[co_pure_field(carrier)]
            + targets[co_at_co2_field(carrier)]
            + targets[co_at_h2o_field(carrier)]
        )
    targets["CO_gas"] = np.maximum(COT - solid_sum, 0.0)

    # CO2 targets.
    CO2T = total_CO2(state)
    fr = params["volatiles"]["CO2"]["solid_fractions"]
    solid_sum = np.zeros_like(CO2T)
    for carrier in CARRIERS:
        fc = float(cf[carrier])
        targets[co2_pure_field(carrier)] = fc * float(fr["pure"]) * ms[carrier]["CO2_pure"] * CO2T
        targets[co2_at_h2o_field(carrier)] = fc * float(fr["at_H2O"]) * ms[carrier]["CO2_at_H2O"] * CO2T
        solid_sum += targets[co2_pure_field(carrier)] + targets[co2_at_h2o_field(carrier)]
    targets["CO2_gas"] = np.maximum(CO2T - solid_sum, 0.0)

    # H2O targets.
    H2OT = total_H2O(state)
    solid_sum = np.zeros_like(H2OT)
    for carrier in CARRIERS:
        fc = float(cf[carrier])
        targets[h2o_field(carrier)] = fc * ms[carrier]["H2O"] * H2OT
        solid_sum += targets[h2o_field(carrier)]
        
    targets["H2O_gas"] = np.maximum(H2OT - solid_sum, 0.0)

    targets = apply_trapping_capacity_limits(targets, params)
    
    return targets

def phase_targets_w_history(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """
    Carrier-history-preserving phase targets.

    This version does NOT repartition the full volatile inventory between
    pebbles and small grains at every phase update.

    Instead:
      - each carrier keeps its own current solid volatile inventory;
      - the shared gas reservoir is partitioned between carriers only for
        possible recondensation, using dust.volatile_carrier_fractions;
      - each carrier's vertically averaged survival fractions determine how
        much of that carrier-associated volatile inventory remains solid;
      - whatever is not assigned to solid reservoirs remains gas.

    This preserves distinct pebble/small-grain transport histories much better
    than the original equilibrium_carrier_fraction version.
    """

    targets = {name: np.zeros_like(next(iter(state.values()))) for name in ALL_FIELDS}

    # Refractory solids do not phase change.
    for carrier in CARRIERS:
        targets[ref_field(carrier)] = state[ref_field(carrier)].copy()

    cf = params["dust"]["volatile_carrier_fractions"]
    ms = vertical["mean_survival"]

    # Normalize carrier gas-condensation weights defensively.
    cf_sum = sum(float(cf[carrier]) for carrier in CARRIERS)
    if cf_sum <= 0.0:
        raise ValueError("dust.volatile_carrier_fractions must have positive sum.")

    w_carrier = {
        carrier: float(cf[carrier]) / cf_sum
        for carrier in CARRIERS
    }

    # ------------------------------------------------------------------
    # CO
    # ------------------------------------------------------------------
    fr = params["volatiles"]["CO"]["solid_fractions"]

    CO_total = total_CO(state)
    CO_solid_target_sum = np.zeros_like(CO_total)

    for carrier in CARRIERS:
        # Existing carrier-specific solid CO inventory.
        CO_solid_carrier = (
            state[co_pure_field(carrier)]
            + state[co_at_co2_field(carrier)]
            + state[co_at_h2o_field(carrier)]
        )

        # Gas is shared. For target construction, assign a fraction of the
        # current gas reservoir to each carrier as the material available for
        # recondensation onto that carrier.
        CO_available_carrier = (
            CO_solid_carrier + \
            w_carrier[carrier] * state["CO_gas"]
        )

        targets[co_pure_field(carrier)] = (
            float(fr["pure"]) * \
            ms[carrier]["CO_pure"] * \
            CO_available_carrier
        )

        targets[co_at_co2_field(carrier)] = (
            float(fr["at_CO2"]) * \
            ms[carrier]["CO_at_CO2"] * \
            CO_available_carrier
        )

        targets[co_at_h2o_field(carrier)] = (
            float(fr["at_H2O"]) * \
            ms[carrier]["CO_at_H2O"] * \
            CO_available_carrier
        )

        CO_solid_target_sum += (
            targets[co_pure_field(carrier)] + \
            targets[co_at_co2_field(carrier)] + \
            targets[co_at_h2o_field(carrier)]
        )

    targets["CO_gas"] = np.maximum(CO_total - CO_solid_target_sum, 0.0)

    # ------------------------------------------------------------------
    # CO2
    # ------------------------------------------------------------------
    fr = params["volatiles"]["CO2"]["solid_fractions"]

    CO2_total = total_CO2(state)
    CO2_solid_target_sum = np.zeros_like(CO2_total)

    for carrier in CARRIERS:
        CO2_solid_carrier = (
            state[co2_pure_field(carrier)]
            + state[co2_at_h2o_field(carrier)]
        )

        CO2_available_carrier = (
            CO2_solid_carrier
            + w_carrier[carrier] * state["CO2_gas"]
        )

        targets[co2_pure_field(carrier)] = (
            float(fr["pure"])
            * ms[carrier]["CO2_pure"]
            * CO2_available_carrier
        )

        targets[co2_at_h2o_field(carrier)] = (
            float(fr["at_H2O"])
            * ms[carrier]["CO2_at_H2O"]
            * CO2_available_carrier
        )

        CO2_solid_target_sum += (
            targets[co2_pure_field(carrier)]
            + targets[co2_at_h2o_field(carrier)]
        )

    targets["CO2_gas"] = np.maximum(CO2_total - CO2_solid_target_sum, 0.0)

    # ------------------------------------------------------------------
    # H2O
    # ------------------------------------------------------------------
    H2O_total = total_H2O(state)
    H2O_solid_target_sum = np.zeros_like(H2O_total)

    for carrier in CARRIERS:
        H2O_solid_carrier = state[h2o_field(carrier)]

        H2O_available_carrier = (
            H2O_solid_carrier
            + w_carrier[carrier] * state["H2O_gas"]
        )

        targets[h2o_field(carrier)] = (
            ms[carrier]["H2O"]
            * H2O_available_carrier
        )

        H2O_solid_target_sum += targets[h2o_field(carrier)]

    targets["H2O_gas"] = np.maximum(H2O_total - H2O_solid_target_sum, 0.0)
    
    targets = apply_trapping_capacity_limits(targets, params)

    return targets


def set_phase_equilibrium(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    
    if params["dust"]["phase_partition"]["preserve_carrier_history"]:
        targets = phase_targets_w_history(state, vertical, params)
    else:
        targets = phase_targets(state, vertical, params)
    for name in ALL_FIELDS:
        state[name] = targets[name]
    return state


def phase_relaxation_step(
    state: Dict[str, np.ndarray],
    dt: float,
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    tau = float(params["volatiles"]["phase_relaxation_time_yr"]) * YR
    fac = math.exp(-dt / tau)
    
    if params["dust"]["phase_partition"]["preserve_carrier_history"]:
        targets = phase_targets_w_history(state, vertical, params)
    else:
        targets = phase_targets(state, vertical, params)

    for name in ALL_FIELDS:
        if name.startswith("ref_solid"):
            continue
        state[name] = targets[name] + (state[name] - targets[name]) * fac

    return state


# -----------------------------
# Finite-volume transport
# -----------------------------
def interp_edges(arr: np.ndarray) -> np.ndarray:
    edge = np.empty(arr.size + 1, dtype=arr.dtype)
    edge[0] = arr[0]
    edge[-1] = arr[-1]
    edge[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    return edge

def update_gas_surface_density(
    disk: Dict[str, np.ndarray],
    dt: float,
    grid: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:

    Sigma_g_old = disk["Sigma_g"]

    rhs, _ = transport_rhs(
        Sigma_g_old,
        disk["v_g"],
        disk["D_g"],
        Sigma_g_old,
        grid,
        diffusion_on=False,
    )

    Sigma_g_new = Sigma_g_old + dt * rhs
    floor = float(params["gas"].get("Sigma_floor", 1.0e-30))
    disk["Sigma_g"] = np.maximum(Sigma_g_new, floor)

    return disk

def refresh_disk_after_sigma_update(
    grid,
    disk,
    params,
):
    Sigma_g = disk["Sigma_g"]
    r = grid["r"]

    gas = params["gas"]
    T_mid = disk["T_mid"]
    H = disk["H"]
    c_s = disk["c_s"]

    rho_mid = Sigma_g / (np.sqrt(2.0 * np.pi) * H)
    P_mid = rho_mid * c_s**2
    dlnP_dlnr = np.gradient(np.log(P_mid), np.log(r), edge_order=2)

    eta = -0.5 * (c_s / disk["v_K"])**2 * dlnP_dlnr

    # If you want v_g prescribed by alpha viscosity:
    nu = disk["nu"]

    if gas["velocity_mode"] == "viscous":
        # Simple prescribed inflow approximation.
        v_g = -1.5 * nu / r
    elif gas["velocity_mode"] == "zero":
        v_g = np.zeros_like(r)
    elif gas["velocity_mode"] == "constant":
        v_g = np.full_like(r, float(gas["constant_v_g_cm_s"]))
    else:
        raise ValueError("Unknown velocity_mode")

    D_g = float(gas["alpha"]) * c_s * H / float(gas["schmidt_gas"])

    disk["rho_mid"] = rho_mid
    disk["P_mid"] = P_mid
    disk["dlnP_dlnr"] = dlnP_dlnr
    disk["eta"] = eta
    disk["v_g"] = v_g
    disk["v_g_visc"] = v_g.copy() # avoid double counting backreaction
    disk["D_g"] = D_g

    return disk

def transport_rhs(
    Sigma: np.ndarray,
    v: np.ndarray,
    D: np.ndarray,
    Sigma_g: np.ndarray,
    grid: Dict[str, np.ndarray],
    diffusion_on: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return dSigma/dt and edge flux Phi = v Sigma - D Sigma_g d(Sigma/Sigma_g)/dr.

    Phi has units g cm^-1 s^-1.
    """
    r = grid["r"]
    r_edge = grid["r_edge"]
    area = grid["area"]
    n = Sigma.size

    v_e = interp_edges(v)

    left = np.empty(n + 1, dtype=np.float64)
    right = np.empty(n + 1, dtype=np.float64)

    # Outflow boundaries: zero external inflow.
    left[0] = 0.0
    left[1:] = Sigma
    right[:-1] = Sigma
    right[-1] = 0.0

    Sigma_up = np.where(v_e >= 0.0, left, right)
    flux = v_e * Sigma_up

    if diffusion_on:
        D_e = interp_edges(D)
        Sigma_g_e = interp_edges(Sigma_g)
        c = Sigma / np.maximum(Sigma_g, 1.0e-300)

        dcdr = np.zeros(n + 1, dtype=np.float64)
        dcdr[1:-1] = (c[1:] - c[:-1]) / (r[1:] - r[:-1])
        # Zero-gradient diffusive boundary flux.
        flux -= D_e * Sigma_g_e * dcdr

    dMdt = -2.0 * PI * (r_edge[1:] * flux[1:] - r_edge[:-1] * flux[:-1])
    dSigdt = dMdt / area
    return dSigdt, flux


def compute_timestep(
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    params: Dict[str, Any],
) -> float:
    num = params["numerics"]
    dr = grid["dr"]

    vmax = np.abs(disk["v_g"]).copy()
    for carrier in CARRIERS:
        vmax = np.maximum(vmax, np.abs(carrier_coeff[carrier]["v"]))

    mask_v = vmax > 0.0
    if np.any(mask_v):
        dt_adv = float(num["cfl_advective"]) * float(np.min(dr[mask_v] / vmax[mask_v]))
    else:
        dt_adv = np.inf

    Dmax = disk["D_g"].copy()
    for carrier in CARRIERS:
        if bool(params["dust"]["carriers"][carrier]["include_diffusion"]):
            Dmax = np.maximum(Dmax, carrier_coeff[carrier]["D"])

    mask_D = Dmax > 0.0
    if np.any(mask_D):
        dt_diff = float(num["cfl_diffusive"]) * float(np.min(dr[mask_D] ** 2 / Dmax[mask_D]))
    else:
        dt_diff = np.inf

    dt_max = float(num["max_timestep_yr"]) * YR
    dt = min(dt_adv, dt_diff, dt_max)
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Computed invalid timestep.")
    return dt


def transport_step(
    state: Dict[str, np.ndarray],
    dt: float,
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    Sigma_g = disk["Sigma_g"]

    new_state = {k: v.copy() for k, v in state.items()}

    vapor_diffusion = bool(params["volatiles"]["include_vapor_diffusion"])
    for name in VAPOR_FIELDS:
        rhs, _ = transport_rhs(
            state[name],
            disk["v_g"],
            disk["D_g"],
            Sigma_g,
            grid,
            diffusion_on=vapor_diffusion,
        )
        new_state[name] = state[name] + dt * rhs

    for carrier in CARRIERS:
        diffusion_on = bool(params["dust"]["carriers"][carrier]["include_diffusion"])
        for name in solid_fields_for(carrier):
            rhs, _ = transport_rhs(
                state[name],
                carrier_coeff[carrier]["v"],
                carrier_coeff[carrier]["D"],
                Sigma_g,
                grid,
                diffusion_on=diffusion_on,
            )
            new_state[name] = state[name] + dt * rhs

    if bool(params["numerics"]["clip_negative"]):
        floor = float(params["numerics"]["surface_density_floor"])
        tol = float(params["numerics"]["negative_tolerance"])
        for name in ALL_FIELDS:
            arr = new_state[name]
            minval = float(np.min(arr))
            if minval < -max(tol, 1.0e-12 * max(1.0, float(np.max(np.abs(arr))))):
                print(
                    f"WARNING: clipping negative values in {name}; min={minval:.3e}. "
                    "Consider reducing CFL or max_timestep.",
                    file=sys.stderr,
                )
            new_state[name] = np.maximum(arr, floor)

    return new_state


# -----------------------------
# Diagnostics and outputs
# -----------------------------
def carbon_oxygen_columns(state: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    CO_s_peb = (
        state[co_pure_field("pebble")]
        + state[co_at_co2_field("pebble")]
        + state[co_at_h2o_field("pebble")]
    )
    CO_s_small = (
        state[co_pure_field("small")]
        + state[co_at_co2_field("small")]
        + state[co_at_h2o_field("small")]
    )
    CO2_s_peb = state[co2_pure_field("pebble")] + state[co2_at_h2o_field("pebble")]
    CO2_s_small = state[co2_pure_field("small")] + state[co2_at_h2o_field("small")]
    H2O_s_peb = state[h2o_field("pebble")]
    H2O_s_small = state[h2o_field("small")]

    nCO_peb = CO_s_peb / MOLECULAR_WEIGHTS["CO"]
    nCO_small = CO_s_small / MOLECULAR_WEIGHTS["CO"]
    nCO2_peb = CO2_s_peb / MOLECULAR_WEIGHTS["CO2"]
    nCO2_small = CO2_s_small / MOLECULAR_WEIGHTS["CO2"]
    nH2O_peb = H2O_s_peb / MOLECULAR_WEIGHTS["H2O"]
    nH2O_small = H2O_s_small / MOLECULAR_WEIGHTS["H2O"]

    nCO_g = state["CO_gas"] / MOLECULAR_WEIGHTS["CO"]
    nCO2_g = state["CO2_gas"] / MOLECULAR_WEIGHTS["CO2"]
    nH2O_g = state["H2O_gas"] / MOLECULAR_WEIGHTS["H2O"]

    C_peb = nCO_peb + nCO2_peb
    O_peb = nCO_peb + 2.0 * nCO2_peb + nH2O_peb
    C_small = nCO_small + nCO2_small
    O_small = nCO_small + 2.0 * nCO2_small + nH2O_small
    C_gas = nCO_g + nCO2_g
    O_gas = nCO_g + 2.0 * nCO2_g + nH2O_g

    C_solid = C_peb + C_small
    O_solid = O_peb + O_small
    eps = 1.0e-300

    return {
        "CO_solid_total": CO_s_peb + CO_s_small,
        "CO_solid_pebble": CO_s_peb,
        "CO_solid_small": CO_s_small,
        "CO_hidden_total": (
            state[co_at_co2_field("pebble")]
            + state[co_at_h2o_field("pebble")]
            + state[co_at_co2_field("small")]
            + state[co_at_h2o_field("small")]
        ),
        "CO2_solid_total": CO2_s_peb + CO2_s_small,
        "CO2_solid_pebble": CO2_s_peb,
        "CO2_solid_small": CO2_s_small,
        "H2O_solid_total": H2O_s_peb + H2O_s_small,
        "H2O_solid_pebble": H2O_s_peb,
        "H2O_solid_small": H2O_s_small,
        "C_over_O_gas": C_gas / np.maximum(O_gas, eps),
        "C_over_O_solid": C_solid / np.maximum(O_solid, eps),
        "C_over_O_pebble": C_peb / np.maximum(O_peb, eps),
        "C_over_O_small": C_small / np.maximum(O_small, eps),
        "hidden_CO_fraction": (
            (
                state[co_at_co2_field("pebble")]
                + state[co_at_h2o_field("pebble")]
                + state[co_at_co2_field("small")]
                + state[co_at_h2o_field("small")]
            )
            / np.maximum(total_CO(state), eps)
        ),
        "gas_CO_fraction": state["CO_gas"] / np.maximum(total_CO(state), eps),
    }


def write_snapshot_1d(
    output_dir: Path,
    snap_idx: int,
    t: float,
    state: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    vertical: Dict[str, Any],
    release: Dict[str, np.ndarray],
) -> None:
    co = carbon_oxygen_columns(state)
    cols: Dict[str, np.ndarray] = {
        "time_yr": np.full_like(grid["r_au"], t / YR),
        "r_au": grid["r_au"],
        "Sigma_g": disk["Sigma_g"],
        "T_mid_K": disk["T_mid"],
        "H_over_r": disk["H"] / grid["r"],
        "v_g_cm_s": disk["v_g"],
        "eta": disk["eta"],
        "D_g_cm2_s": disk["D_g"],
    }

    for carrier in CARRIERS:
        cols[f"St_{carrier}"] = carrier_coeff[carrier]["St"]
        cols[f"v_{carrier}_cm_s"] = carrier_coeff[carrier]["v"]
        cols[f"D_{carrier}_cm2_s"] = carrier_coeff[carrier]["D"]
        cols[f"Hd_over_H_{carrier}"] = carrier_coeff[carrier]["H_ratio"]

    for key in SURVIVAL_KEYS:
        cols[f"snow_surface_z_over_r_{key}"] = vertical["snow_surfaces_z_over_r"][key]
        for carrier in CARRIERS:
            cols[f"mean_survival_{key}_{carrier}"] = vertical["mean_survival"][carrier][key]

    for name in ALL_FIELDS:
        cols[name] = state[name]
        
    for key, arr in release.items():
        if arr is not None:
            cols[key] = arr
        else:
            cols[key] = np.zeros_like(grid["r_au"])
        
    cols.update(co)

    names = list(cols.keys())
    arr = np.column_stack([cols[k] for k in names])
    path = output_dir / "snapshots" / f"snapshot_{snap_idx:06d}.csv"
    np.savetxt(path, arr, delimiter=",", header=",".join(names), comments="")


def distribute_surface_density_rz(
    Sigma: np.ndarray,
    weights: np.ndarray,
    modifier: np.ndarray | None = None,
) -> np.ndarray:
    """Distribute a radial surface density over vertical bins.

    weights and modifier have shape (n_r, n_z). Output has shape (n_r, n_z)
    and sums over z bins to Sigma at each radius.
    """
    if modifier is None:
        shape = weights.copy()
    else:
        shape = weights * np.maximum(modifier, 0.0)

    norm = np.sum(shape, axis=1)
    out = np.zeros_like(shape)
    ok = norm > 1.0e-300
    out[ok, :] = Sigma[ok, None] * shape[ok, :] / norm[ok, None]
    # Fallback if survival is zero but a tiny amount remains due to finite relaxation.
    if np.any(~ok):
        fallback = weights[~ok, :]
        fallback_norm = np.sum(fallback, axis=1)
        out[~ok, :] = Sigma[~ok, None] * fallback / np.maximum(fallback_norm[:, None], 1.0e-300)
    return out


def write_snapshot_2d(
    output_dir: Path,
    snap_idx: int,
    t: float,
    state: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    vertical: Dict[str, Any],
) -> None:
    data: Dict[str, Any] = {
        "time_yr": np.array(t / YR),
        "r_au": grid["r_au"],
        "y_z_over_H": vertical["y"],
        "z_au": vertical["z_au"],
        "z_over_r": vertical["z_over_r"],
        "T_K": vertical["T_rz"],
        "Av": vertical["Av"],
        "shield": vertical["shield"],
        "Sigma_g": disk["Sigma_g"],
    }

    for key in SURVIVAL_KEYS:
        data[f"survival_{key}"] = vertical["survival"][key]
        data[f"snow_surface_z_over_r_{key}"] = vertical["snow_surfaces_z_over_r"][key]

    # Gas vapor surface-density bins.
    w_gas = vertical["weights"]["gas"]
    for name in VAPOR_FIELDS:
        data[f"surfbin_{name}"] = distribute_surface_density_rz(state[name], w_gas)

    # Solid carrier surface-density bins.
    for carrier in CARRIERS:
        w = vertical["weights"][carrier]
        data[f"weight_{carrier}"] = w
        for name in solid_fields_for(carrier):
            if name.startswith("ref_solid"):
                data[f"surfbin_{name}"] = distribute_surface_density_rz(state[name], w)
            else:
                key = FIELD_TO_SURVIVAL_KEY[name]
                data[f"surfbin_{name}"] = distribute_surface_density_rz(
                    state[name], w, vertical["survival"][key]
                )

    path = output_dir / "snapshots_2d" / f"snapshot2d_{snap_idx:06d}.npz"
    np.savez_compressed(path, **data)


def initialize_diagnostics(path: Path) -> None:
    fieldnames = [
        "snapshot",
        "step",
        "time_yr",
        "dt_yr",
        "M_gas_fixed_msun",
        "M_ref_pebble_mearth",
        "M_ref_small_mearth",
        "M_CO_total_mearth",
        "M_CO_gas_mearth",
        "M_CO_solid_mearth",
        "M_CO_hidden_mearth",
        "M_CO_pebble_mearth",
        "M_CO_small_mearth",
        "M_CO2_total_mearth",
        "M_CO2_gas_mearth",
        "M_CO2_solid_mearth",
        "M_CO2_pebble_mearth",
        "M_CO2_small_mearth",
        "M_H2O_total_mearth",
        "M_H2O_gas_mearth",
        "M_H2O_solid_mearth",
        "M_H2O_pebble_mearth",
        "M_H2O_small_mearth",
        "min_sigma",
        "max_sigma",
        "max_epsilon_pebble",
        "median_epsilon_pebble",
        "max_backreaction_X",
        "median_backreaction_X",
        "max_backreaction_Y",
        "median_backreaction_Y",
        "max_backreaction_A",
        "median_backreaction_A",
        "max_backreaction_B",
        "median_backreaction_B",
        "min_vr_g",
        "max_vr_g",
        "min_pebble_v",
        "max_pebble_v",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()


def append_diagnostics(
    path: Path,
    snap_idx: int,
    step: int,
    t: float,
    dt: float,
    state: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    br_diag: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, np.ndarray],
) -> None:
    area = grid["area"]

    def mass_field(name: str) -> float:
        return annulus_integral_sigma(area, state[name])

    def mass_array(arr: np.ndarray) -> float:
        return annulus_integral_sigma(area, arr)

    CO_peb = state[co_pure_field("pebble")] + state[co_at_co2_field("pebble")] + state[co_at_h2o_field("pebble")]
    CO_small = state[co_pure_field("small")] + state[co_at_co2_field("small")] + state[co_at_h2o_field("small")]
    CO_hidden = state[co_at_co2_field("pebble")] + state[co_at_h2o_field("pebble")] + state[co_at_co2_field("small")] + state[co_at_h2o_field("small")]

    CO2_peb = state[co2_pure_field("pebble")] + state[co2_at_h2o_field("pebble")]
    CO2_small = state[co2_pure_field("small")] + state[co2_at_h2o_field("small")]

    H2O_peb = state[h2o_field("pebble")]
    H2O_small = state[h2o_field("small")]

    all_min = min(float(np.min(state[k])) for k in ALL_FIELDS)
    all_max = max(float(np.max(state[k])) for k in ALL_FIELDS)

    row = {
        "snapshot": snap_idx,
        "step": step,
        "time_yr": t / YR,
        "dt_yr": dt / YR,
        "M_gas_fixed_msun": annulus_integral_sigma(area, disk["Sigma_g"]) / MSUN,
        "M_ref_pebble_mearth": mass_field(ref_field("pebble")) / MEARTH,
        "M_ref_small_mearth": mass_field(ref_field("small")) / MEARTH,
        "M_CO_total_mearth": mass_array(total_CO(state)) / MEARTH,
        "M_CO_gas_mearth": mass_field("CO_gas") / MEARTH,
        "M_CO_solid_mearth": mass_array(CO_peb + CO_small) / MEARTH,
        "M_CO_hidden_mearth": mass_array(CO_hidden) / MEARTH,
        "M_CO_pebble_mearth": mass_array(CO_peb) / MEARTH,
        "M_CO_small_mearth": mass_array(CO_small) / MEARTH,
        "M_CO2_total_mearth": mass_array(total_CO2(state)) / MEARTH,
        "M_CO2_gas_mearth": mass_field("CO2_gas") / MEARTH,
        "M_CO2_solid_mearth": mass_array(CO2_peb + CO2_small) / MEARTH,
        "M_CO2_pebble_mearth": mass_array(CO2_peb) / MEARTH,
        "M_CO2_small_mearth": mass_array(CO2_small) / MEARTH,
        "M_H2O_total_mearth": mass_array(total_H2O(state)) / MEARTH,
        "M_H2O_gas_mearth": mass_field("H2O_gas") / MEARTH,
        "M_H2O_solid_mearth": mass_array(H2O_peb + H2O_small) / MEARTH,
        "M_H2O_pebble_mearth": mass_array(H2O_peb) / MEARTH,
        "M_H2O_small_mearth": mass_array(H2O_small) / MEARTH,
        "min_sigma": all_min,
        "max_sigma": all_max,
        "max_epsilon_pebble": np.nanmax(br_diag.get("epsilon_pebble", [0])),
        "median_epsilon_pebble": np.nanmedian(br_diag.get("epsilon_pebble", [0])),
        "max_backreaction_X": np.nanmax(disk.get("backreaction_X", [0])),
        "median_backreaction_X": np.nanmedian(disk.get("backreaction_X", [0])),
        "max_backreaction_Y": np.nanmax(disk.get("backreaction_Y", [0])),
        "median_backreaction_Y": np.nanmedian(disk.get("backreaction_Y", [0])),
        "max_backreaction_A": np.nanmax(disk.get("backreaction_A", [0])),
        "median_backreaction_A": np.nanmedian(disk.get("backreaction_A", [0])),
        "max_backreaction_B": np.nanmax(disk.get("backreaction_B", [0])),
        "median_backreaction_B": np.nanmedian(disk.get("backreaction_B", [0])),
        "min_vr_g": np.nanmin(disk["v_g"]),
        "max_vr_g": np.nanmax(disk["v_g"]),
        "min_pebble_v": np.nanmin(carrier_coeff["pebble"]["v"]),
        "max_pebble_v": np.nanmax(carrier_coeff["pebble"]["v"]),
    }

    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        writer.writerow(row)


# -----------------------------
# Main run
# -----------------------------
def write_outputs(
    output_dir: Path,
    snap_idx: int,
    step: int,
    t: float,
    dt: float,
    state: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    diag_path: Path,
    release: Dict[str, np.ndarray],
    br_diag: Dict[str, np.ndarray],
) -> None:
    if bool(params["output"]["save_1d_csv"]):
        write_snapshot_1d(output_dir, snap_idx, t, state, grid, disk, 
                          carrier_coeff, vertical, release)

    append_diagnostics(diag_path, snap_idx, step, t, dt, state, grid, disk, br_diag, carrier_coeff)

    if bool(params["output"]["save_2d_npz"]):
        every = int(params["output"]["save_2d_every_n_snapshots"])
        if every <= 1 or snap_idx % every == 0:
            write_snapshot_2d(output_dir, snap_idx, t, state, grid, disk, carrier_coeff, vertical)


def run(params: Dict[str, Any]) -> None:
    output_dir = Path(params["simulation"]["output_dir"]).expanduser().resolve()
    ensure_output_dir(
        output_dir,
        overwrite=bool(params["simulation"]["overwrite"]),
        save_2d=bool(params["output"]["save_2d_npz"]),
    )
    write_resolved_params(params, output_dir)

    grid = make_grid(params)
    disk = build_fixed_disk(grid, params)
    carrier_coeff = build_carrier_coefficients(grid, disk, params)
    vertical = make_vertical_structure(grid, disk, carrier_coeff, params)
    state = initialize_state(grid, disk, vertical, params)

    t_end = float(params["simulation"]["t_end_yr"]) * YR
    if params["simulation"]["save_interval_yr"] is None:
        save_interval = float(params["simulation"]["save_interval_fraction"]) * t_end
    else:
        save_interval = float(params["simulation"]["save_interval_yr"]) * YR
    save_interval = max(save_interval, 1.0e-30)

    diag_path = output_dir / "diagnostics.csv"
    initialize_diagnostics(diag_path)
    
    update_gas = bool(params["gas"]["update_gas"])

    t = 0.0
    step = 0
    snap_idx = 0
    next_save = 0.0
    last_dt = 0.0

    release = {"Mdot_dlnr_CO_pure": None,
               "Mdot_dlnr_CO_at_CO2": None,
               "Mdot_dlnr_CO_at_H2O": None,
               "dM_CO_pure": None,
               "dM_CO_at_CO2": None,
               "dM_CO_at_H2O": None,
               "dSigma_CO_gas": None,
               "dM_CO_gas": None,
               "Mdot_CO_gas": None,
               "Mdot_dlnr_CO_gas": None,
               }

    br_diag = {}
    write_outputs(
        output_dir, snap_idx, step, t, last_dt, state, grid, disk, carrier_coeff,
        vertical, params, diag_path, release, br_diag,
    )
    snap_idx += 1
    next_save += save_interval
    
    backreaction = get_backreaction_enabled(params)

    progress_every = int(params["simulation"]["progress_every"])
    print(f"Writing outputs to: {output_dir}")
    print(f"t_end = {t_end / YR:.6g} yr, save_interval = {save_interval / YR:.6g} yr")
    print(
        "1+1D model: radial transport with vertically averaged snow-surface phase terms; "
        "evolving H2/He gas disk."
    )

    while t < t_end * (1.0 - 1.0e-14):
        dt = compute_timestep(grid, disk, carrier_coeff, params)
        dt = min(dt, t_end - t, next_save - t if next_save > t else dt)

        if dt <= 0.0:
            dt = min(compute_timestep(grid, disk, carrier_coeff, params), t_end - t)

            
        br_diag = {}
        if update_gas:
            
            # Apply BR from current solids
            if backreaction:
                # Refresh no-BR quantities from current Sigma_g
                disk = refresh_disk_after_sigma_update(grid, disk, params)
                carrier_coeff = build_carrier_coefficients(grid, disk, params)
                disk, carrier_coeff, br_diag = apply_dust_backreaction_to_velocities(
                    state, disk, carrier_coeff, params
                )
                
            disk = update_gas_surface_density(disk, dt, grid, params)
            
        disk = refresh_disk_after_sigma_update(grid, disk, params)
        carrier_coeff = build_carrier_coefficients(grid, disk, params)
        # Apply backreaction using updated disc
        if backreaction:
            disk, carrier_coeff, br_diag = apply_dust_backreaction_to_velocities(
                                            state,
                                            disk,
                                            carrier_coeff,
                                            params,
                                            )

        vertical = make_vertical_structure(grid, disk, carrier_coeff, params)

        state = transport_step(state, dt, grid, disk, carrier_coeff, params)
        state_before_phase = {k: v.copy() for k, v in state.items()}
        state = phase_relaxation_step(state, dt, vertical, params)
        release = compute_co_release(
            state_before_phase,
            state,
            dt,
            grid,
        )

        t += dt
        step += 1
        last_dt = dt

        if t >= next_save - 1.0e-9 * save_interval or t >= t_end * (1.0 - 1.0e-14):
            write_outputs(
                output_dir, snap_idx, step, t, last_dt, state, grid, disk,
                carrier_coeff, vertical, params, diag_path, release, br_diag,
            )
            snap_idx += 1
            next_save += save_interval

        if progress_every > 0 and step % progress_every == 0:
            print(
                f"step={step:8d} t={t / YR:12.5e} yr "
                f"dt={last_dt / YR:10.3e} yr snap={snap_idx - 1}",
                flush=True,
            )

    print(f"Done. steps={step}, snapshots={snap_idx}, final time={t / YR:.6g} yr")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fixed-gas 1+1D mixed-ice volatile transport model with two solid carriers."
    )
    parser.add_argument("params", type=str, help="Path to YAML parameter file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    run(params)


if __name__ == "__main__":
    main()
