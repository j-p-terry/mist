"""
mixed_ice_transport_1p1d.py

What this does
--------------
- Radial transport is Eulerian and 1D in surface density.
- Bulk gas evolves by conservative viscous diffusion (default) with optional
  dust backreaction; legacy prescribed-inflow advection remains selectable.
- Optional stationary Gaussian viscosity enhancement creates a gas gap and
  (if strong enough) a pressure maximum at its outer edge; its initial gas
  depression and dust diffusion treatment are separately configurable.
- Viscous diffusion is implicit; pressure-feedback transport is explicitly
  CFL limited. Gas-phase volatiles share the exact bulk-gas interface flux.
- Phase changes are informed by a prescribed vertical structure T(r,z),
    shielding A_V(r,z), and carrier-dependent dust scale heights.
- In history-preserving runs, rapid midplane thermal release and slow
    vertical exposure/mixing are integrated using physical elapsed time,
    never by repeatedly multiplying existing ice by mean survival.
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
- Optional no-resequestration sensitivity: inherited matrix-associated ice can
  sublimate, but released vapor cannot re-enter a mixed reservoir; it may only
  remain gas or recondense as pure ice where thermally allowed.

The "1+1D" approximation
------------------------
The evolved variables are radial surface densities Sigma_i(r,t). At every radius
compute a vertical disk column, snow surfaces, and carrier-weighted survival
fractions. Those vertically averaged survival fractions determine the phase
source terms. Optional 2D snapshot files reconstruct where the reservoirs would
sit vertically, but the transport itself remains 1D radial.

Usage
-----
    python mixed_ice_transport_1p1d.py example_mist_params.yaml

Outputs
-------
    <output_dir>/snapshots/snapshot_XXXXXX.csv
    <output_dir>/snapshots_2d/snapshot2d_XXXXXX.npz        [optional]
    <output_dir>/diagnostics.csv
    <output_dir>/resolved_params.yaml

Release diagnostics
-------------------
Each saved 1D snapshot contains release integrated over the interval since the
previous saved snapshot (``dM_*``/``dSigma_*``), plus output-cadence-independent
running integrals (``cum_dM_*``/``cum_dSigma_*``). Reservoir-channel release is
a gross positive loss from that ice reservoir during the phase update; net gas
phase source terms are stored separately.

Additional runtime diagnostics
------------------------------
The model also accumulates inner/outer boundary losses, gross host-capacity
rejection, capacity-rejected material routed to gas, positivity corrections,
local phase-conservation errors, timestep statistics, and backreaction strength.
These quantities are written to diagnostics.csv; per-radius capacity diagnostics
are also included in the 1D snapshots.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shutil
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

# Reservoir fields used by the release bookkeeping.  These diagnostics measure
# gross positive loss from each reservoir during the phase-update substep.
ICE_RELEASE_CHANNEL_FIELDS = {
    "CO_pure": ("CO_pure_ice_pebble", "CO_pure_ice_small"),
    "CO_at_CO2": ("CO_at_CO2_ice_pebble", "CO_at_CO2_ice_small"),
    "CO_at_H2O": ("CO_at_H2O_ice_pebble", "CO_at_H2O_ice_small"),
    "CO2_pure": ("CO2_pure_ice_pebble", "CO2_pure_ice_small"),
    "CO2_at_H2O": ("CO2_at_H2O_ice_pebble", "CO2_at_H2O_ice_small"),
    "H2O_pure": ("H2O_ice_pebble", "H2O_ice_small"),
}

PHASE_GAS_FIELDS = {
    "CO": "CO_gas",
    "CO2": "CO2_gas",
    "H2O": "H2O_gas",
}


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

# Molecular species carried by each evolved field. Refractory fields enter
# carrier and boundary budgets but not volatile-species conservation balances.
FIELD_TO_SPECIES: Dict[str, str] = {
    "CO_gas": "CO",
    "CO2_gas": "CO2",
    "H2O_gas": "H2O",
}
for _carrier in CARRIERS:
    FIELD_TO_SPECIES[ref_field(_carrier)] = "refractory"
    FIELD_TO_SPECIES[h2o_field(_carrier)] = "H2O"
    FIELD_TO_SPECIES[co2_pure_field(_carrier)] = "CO2"
    FIELD_TO_SPECIES[co2_at_h2o_field(_carrier)] = "CO2"
    FIELD_TO_SPECIES[co_pure_field(_carrier)] = "CO"
    FIELD_TO_SPECIES[co_at_co2_field(_carrier)] = "CO"
    FIELD_TO_SPECIES[co_at_h2o_field(_carrier)] = "CO"


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
        "t_end_yr": 2.5e5,
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
        # Whether the bulk H2/He gas evolves at all.
        "update_gas": True,
        # viscous_diffusion (default): Pringle thin-disk diffusion with the
        # Gárate-type dust-backreaction modification to the gas mass flux.
        # prescribed_advection: legacy advect Sigma_g with the locally prescribed
        # velocity (-3nu/2r before backreaction).
        "evolution_mode": "viscous_diffusion",
        # Outflow inner edge avoids imposing an artificially rapid zero-torque
        # drain in the first radial cell. Optional: zero_torque.
        # Viscous flux is zero at the outer edge. Dust-feedback flow can leave
        # either edge, but not enter from outside.
        "viscous_inner_boundary": "outflow",
        # Persistent, stationary viscosity enhancement representing a gas gap.
        # This is a prescribed disk substructure, not a planet torque.
        "gap": {
            "enabled": False,
            "center_au": 20.0,
            "width_mode": "scale_height",  # scale_height or au
            "width_H": 2.0,             # Gaussian sigma in H(r_gap)
            "width_au": 2.0,            # used only when width_mode: au
            "viscosity_factor": 10.0,   # peak alpha_visc / alpha_background
            # inverse_viscosity starts with a pre-formed gas gap: Sigma_g ~ 1/alpha_visc
            # smooth starts with the unperturbed Sigma_g and lets the gap emerge.
            "initial_profile": "inverse_viscosity",  # or smooth
            # Off by default: don't conflate gas angular-momentum transport
            # with dust/vapor turbulent diffusion or vertical stirring.
            "diffusion_follows_viscosity": False,
        },
        "Sigma_floor": 1.0e-30,
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
        # t=0 volatile inventory allocation across pebble/small carriers.
        # These fractions do NOT allocate new condensate in the new closure.
        # They should sum to 1.
        "volatile_carrier_fractions": {
            "pebble": 0.90,
            "small": 0.10,
        },
        # Retain carrier transport histories, with a separate switch to preserve
        # the identities of already-formed ice matrices until sublimation.
        "phase_partition": {
            "preserve_carrier_history": True,
            # Allocation of shared vapor between carriers during recondensation.
            #
            # available_surface_area: distribute new condensate among the
            # surfaces actually able to retain each ice reservoir, proportional
            # to [Sigma_ref/(rho_s*a)] * f_reservoir * mean_survival.
            # Total condensed vapor is separately bounded by the configured
            # capture closure; do not multiply deposition by survival twice.
            # surface_area and fixed retain older *carrier weighting* closures
            # for sensitivity tests, not bit-for-bit reproductions of the old
            # complete phase algorithm.
            "condensation_weighting": "available_surface_area",
            "condensation_area_floor": 1.0e-300,
            "condensation_survival_floor": 1.0e-10,
            # An efficient vertical mixing closure, NOT a freeze-out kinetic law.
            "condensation_capture_mode": "max_survival",
            # Avoid proposing new guest deposition on an already full matrix.
            "prelimit_condensation_by_capacity": True,
            # Retain the host identity of extant ice until it releases to gas.
            "preserve_matrix_history": True,
            # If False, inherited matrix-associated reservoirs may only be lost;
            # vapor can recondense as pure ice but cannot be re-sequestered into
            # CO@CO2, CO@H2O, or CO2@H2O after t=0.
            "allow_resequestration": True,
            # Loss of pre-existing pure and matrix-associated ices:
            # midplane thermal instability is relaxed rapidly; exposure to
            # warm/unshielded upper layers is processed over a physical
            # vertical-mixing time, NOT once per numerical timestep.
            # 'thermal_only' disables the slow vertical-processing channel.
            "ice_loss_mode": "mixing_limited",  # mixing_limited or thermal_only
            "vertical_mixing_timescale_factor": 1.0,  # t_mix=factor/(alpha_z*Omega)
            "vertical_exposure_efficiency": 1.0,  # 0..1, uncertain processing efficiency
        },
        "backreaction": {
            "enabled": False,
            "epsilon_mode": "midplane",
            "include_small": False,
            "epsilon_cap": 10.0,
            "sigma_floor": 1.0e-300,
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
                "at_H2O": 150.0,
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
                "at_H2O": 150.0,
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
        "trapping_capacity": {
            "enabled": True,
            "excess_destination": "gas",
            "host_floor": 1.0e-300,
            "availability_relative_floor": 1.0e-12,
            "availability_abs_floor": 1.0e-300,
            "CO_at_CO2": {
                "max_guest_per_host_mol": 0.5,
                "host_mode": "pure_CO2_ice",
            },
            "CO_at_H2O": {"max_guest_per_host_mol": 0.25},
            "CO2_at_H2O": {"max_guest_per_host_mol": 0.5},
            "H2O_total_guest_capacity": {
                "max_total_guest_per_host_mol": None,
            },
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
        # Phase exchange should conserve each molecular species locally.
        "check_phase_conservation": True,
        "phase_conservation_rtol": 1.0e-10,
        "phase_conservation_atol": 1.0e-30,
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
    if p["gas"].get("evolution_mode", "viscous_diffusion") not in {
        "viscous_diffusion", "prescribed_advection"
    }:
        raise ValueError("gas.evolution_mode must be viscous_diffusion or prescribed_advection")
    if (p["gas"].get("update_gas", True)
        and p["gas"]["evolution_mode"] == "viscous_diffusion"
        and p["gas"]["velocity_mode"] != "viscous"):
        raise ValueError("gas.evolution_mode=viscous_diffusion requires gas.velocity_mode=viscous")
    if p["gas"].get("viscous_inner_boundary", "outflow") not in {"outflow", "zero_torque"}:
        raise ValueError("gas.viscous_inner_boundary must be 'outflow' or 'zero_torque'")

    # The Gaussian alpha enhancement must be applied by the *viscous* gas solver;
    # otherwise it cannot maintain a self-consistent gap or pressure maximum.
    gap = p["gas"].get("gap", {})
    if not isinstance(gap, dict):
        raise ValueError("gas.gap must be a mapping")
    if not isinstance(gap.get("enabled", False), bool):
        raise ValueError("gas.gap.enabled must be boolean")
    if gap.get("enabled", False):
        if not p["gas"].get("update_gas", True) or p["gas"].get("evolution_mode") != "viscous_diffusion":
            raise ValueError("gas.gap.enabled requires update_gas: true and evolution_mode: viscous_diffusion")
        if p["gas"].get("velocity_mode") != "viscous":
            raise ValueError("gas.gap.enabled requires velocity_mode: viscous")
        center = float(gap.get("center_au", 20.0))
        if not float(p["grid"]["r_in_au"]) < center < float(p["grid"]["r_out_au"]):
            raise ValueError("gas.gap.center_au must lie inside the radial grid")
        if gap.get("width_mode", "scale_height") not in {"scale_height", "au"}:
            raise ValueError("gas.gap.width_mode must be 'scale_height' or 'au'")
        if gap.get("width_mode", "scale_height") == "scale_height":
            if float(gap.get("width_H", 2.0)) <= 0:
                raise ValueError("gas.gap.width_H must be positive")
        elif float(gap.get("width_au", 2.0)) <= 0:
            raise ValueError("gas.gap.width_au must be positive")
        if not np.isfinite(float(gap.get("viscosity_factor", 10.0))) or float(gap.get("viscosity_factor", 10.0)) < 1.0:
            raise ValueError("gas.gap.viscosity_factor must be finite and >= 1")
        if gap.get("initial_profile", "inverse_viscosity") not in {"smooth", "inverse_viscosity"}:
            raise ValueError("gas.gap.initial_profile must be 'smooth' or 'inverse_viscosity'")
        if not isinstance(gap.get("diffusion_follows_viscosity", False), bool):
            raise ValueError("gas.gap.diffusion_follows_viscosity must be boolean")

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

    phase_partition_cfg = p["dust"]["phase_partition"]
    if phase_partition_cfg.get("preserve_carrier_history") not in {True, False}:
        raise ValueError("dust.phase_partition.preserve_carrier_history must be boolean.")
    if phase_partition_cfg.get("allow_resequestration", True) not in {True, False}:
        raise ValueError("dust.phase_partition.allow_resequestration must be boolean.")
    if phase_partition_cfg.get("ice_loss_mode", "mixing_limited") not in {"mixing_limited", "thermal_only"}:
        raise ValueError("dust.phase_partition.ice_loss_mode must be mixing_limited or thermal_only")
    if not np.isfinite(float(phase_partition_cfg.get("vertical_mixing_timescale_factor", 1.0))) or float(phase_partition_cfg.get("vertical_mixing_timescale_factor", 1.0)) <= 0.0:
        raise ValueError("vertical_mixing_timescale_factor must be finite and > 0")
    if not 0.0 <= float(phase_partition_cfg.get("vertical_exposure_efficiency", 1.0)) <= 1.0:
        raise ValueError("vertical_exposure_efficiency must be in [0, 1]")
    if not 0.0 <= float(p["vertical"].get("alpha_z", 0.001)):
        raise ValueError("vertical.alpha_z must be nonnegative")

    condensation_weighting = str(
        phase_partition_cfg.get("condensation_weighting", "surface_area")
    ).lower()
    if condensation_weighting not in {"available_surface_area", "surface_area", "fixed"}:
        raise ValueError("condensation_weighting must be available_surface_area, surface_area, or fixed")
    if str(phase_partition_cfg.get("condensation_capture_mode", "max_survival")) not in {
        "max_survival", "area_weighted_survival"
    }:
        raise ValueError("condensation_capture_mode must be max_survival or area_weighted_survival")
    if not 0 <= float(phase_partition_cfg.get("condensation_survival_floor", 1e-10)) < 1:
        raise ValueError("condensation_survival_floor must be in [0,1)")
    if phase_partition_cfg.get("preserve_matrix_history", True) not in {True, False}:
        raise ValueError("preserve_matrix_history must be boolean")
    if phase_partition_cfg.get("prelimit_condensation_by_capacity", True) not in {True, False}:
        raise ValueError("prelimit_condensation_by_capacity must be boolean")
    if condensation_weighting == "available_surface_area" and not bool(phase_partition_cfg["preserve_carrier_history"]):
        raise ValueError("available_surface_area requires preserve_carrier_history: true")
    if float(phase_partition_cfg.get("condensation_area_floor", 1.0e-300)) < 0.0:
        raise ValueError("dust.phase_partition.condensation_area_floor must be >= 0.")

    if condensation_weighting in {"available_surface_area", "surface_area"}:
        for carrier in CARRIERS:
            ccfg = p["dust"]["carriers"][carrier]
            if float(ccfg["grain_size_cm"]) <= 0.0:
                raise ValueError(
                    f"dust.carriers.{carrier}.grain_size_cm must be > 0 when "
                    "condensation_weighting='surface_area'."
                )
            if float(ccfg["rho_s_g_cm3"]) <= 0.0:
                raise ValueError(
                    f"dust.carriers.{carrier}.rho_s_g_cm3 must be > 0 when "
                    "condensation_weighting='surface_area'."
                )

    if (
        not bool(phase_partition_cfg.get("allow_resequestration", True))
        and not bool(phase_partition_cfg.get("preserve_matrix_history", True))
    ):
        raise ValueError("No-resequestration requires preserve_matrix_history: true")

    if (
        not bool(phase_partition_cfg.get("allow_resequestration", True))
        and not bool(phase_partition_cfg.get("preserve_carrier_history", True))
    ):
        raise ValueError(
            "No-resequestration mode requires "
            "dust.phase_partition.preserve_carrier_history: true, because the "
            "inherited matrix identity must be retained explicitly."
        )

    br = p["dust"].get("backreaction", {})
    if isinstance(br, dict) and br.get("epsilon_mode", "midplane") not in {"midplane", "column"}:
        raise ValueError("dust.backreaction.epsilon_mode must be 'midplane' or 'column'.")

    if bool(p["vertical"]["enabled"]):
        if int(p["vertical"]["n_z"]) < 8:
            raise ValueError("vertical.n_z should be at least 8.")
        if float(p["vertical"]["zmax_H"]) <= 0:
            raise ValueError("vertical.zmax_H must be positive.")

    if p["simulation"]["save_interval_yr"] is None:
        frac = float(p["simulation"]["save_interval_fraction"])
        if frac <= 0:
            raise ValueError("simulation.save_interval_fraction must be positive.")
    elif float(p["simulation"]["save_interval_yr"]) <= 0:
        raise ValueError("simulation.save_interval_yr must be positive when supplied.")

    if int(p["output"]["save_2d_every_n_snapshots"]) < 1:
        raise ValueError("output.save_2d_every_n_snapshots must be at least 1.")


def ensure_output_dir(path: Path, overwrite: bool, save_2d: bool) -> None:
    """Prepare an output directory without leaving stale model snapshots.

    Earlier versions overwrote low-numbered snapshots in place but could leave
    old high-numbered files behind, causing later analysis to mix two runs.
    With ``overwrite=True`` only model-generated files/directories are cleared;
    unrelated material in the output directory is left untouched.
    """
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(
            f"Output directory {path} exists and is non-empty. "
            "Set simulation.overwrite: true or choose a new output_dir."
        )

    path.mkdir(parents=True, exist_ok=True)
    if overwrite:
        for dirname in ("snapshots", "snapshots_2d"):
            target = path / dirname
            if target.exists():
                shutil.rmtree(target)
        for filename in ("diagnostics.csv", "resolved_params.yaml"):
            target = path / filename
            if target.exists():
                target.unlink()

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


def gaussian_viscosity_gap(
    grid: Dict[str, np.ndarray],
    params: Dict[str, Any],
    H: np.ndarray,
) -> Tuple[np.ndarray, float]:
    """Stationary Gaussian *viscosity enhancement* and its width [cm].

    alpha_visc(r) = alpha0 * [1 + (factor-1) * exp(-(r-r_gap)^2/(2*w^2))].
    The perturbation is imposed on angular-momentum transport, not on the
    temperature profile. Its steady response is a depleted gas surface density
    and, for sufficiently strong/narrow gaps, a pressure maximum on the outer
    gap shoulder. The pressure maximum's radius is *not* r_gap.

    Default width is 2 H_g(r_gap); dust turbulent diffusion and vertical
    settling remain tied to their independent coefficients by default.
    """
    r = grid["r"]
    gap = params["gas"].get("gap", {})
    if not gap.get("enabled", False):
        return np.ones_like(r), 0.0

    center_cm = float(gap["center_au"]) * AU
    if gap.get("width_mode", "scale_height") == "au":
        width_cm = float(gap["width_au"]) * AU
    else:
        width_cm = float(gap["width_H"]) * float(np.interp(center_cm, r, H))
    factor = float(gap["viscosity_factor"])
    profile = 1.0 + (factor - 1.0) * np.exp(
        -0.5 * ((r - center_cm) / width_cm) ** 2
    )
    return profile, width_cm


def viscous_inner_outflow_coefficient(
    grid: Dict[str, np.ndarray], nu: np.ndarray, mode: str,
) -> float:
    """Positive cm/s coefficient of Sigma_0 in inner outgoing viscous flux."""
    r0, rin = grid["r"][0], grid["r_edge"][0]
    if mode == "outflow":
        return 1.5 * nu[0] / r0
    if mode == "zero_torque":
        return 3.0 * nu[0] * np.sqrt(r0) / (np.sqrt(rin) * (r0-rin))
    raise ValueError(f"Unknown viscous inner boundary: {mode}")


def viscous_gas_flux_edges(
    Sigma_g: np.ndarray,
    grid: Dict[str, np.ndarray],
    nu: np.ndarray,
    A: Optional[np.ndarray] = None,
    inner_boundary: str = "outflow",
) -> np.ndarray:
    """Physical viscous mass flux per circumference [g cm^-1 s^-1].

    For Keplerian rotation, F_nu = -3 A(r)/sqrt(r) d/dr(nu Sigma sqrt(r)).
    The full viscous gas equation follows from -div(F_nu) when A=1.

    Inner boundary: outflow with v=-3nu/(2r) (default), or zero torque.
    The former avoids a rapid emptying of the first cell on the short
    viscous-diffusion timescale set by its grid-center to edge distance.
    Outer boundary: zero viscous mass flux. No external gas is supplied.
    """
    r = grid["r"]
    re = grid["r_edge"]
    Q = nu * Sigma_g * np.sqrt(r)
    coeff = np.ones_like(re) if A is None else interp_edges(A)
    out = np.zeros_like(re)
    out[1:-1] = -3.0 * coeff[1:-1] / np.sqrt(re[1:-1]) * (
        (Q[1:] - Q[:-1]) / (r[1:] - r[:-1])
    )
    out[0] = -(coeff[0] * viscous_inner_outflow_coefficient(grid, nu, inner_boundary)
               * Sigma_g[0])
    out[-1] = 0.0
    return out


def gas_viscous_velocity_from_flux(
    Sigma_g: np.ndarray,
    grid: Dict[str, np.ndarray],
    nu: np.ndarray,
    inner_boundary: str = "outflow",
) -> np.ndarray:
    """Cell gas viscous velocity consistent with the face-flux geometry.

    The exact viscous *gas mass* evolution uses the face flux directly; this
    cell-centered velocity is used for pebble drift, vapor transport when gas
    is fixed, and computing the dust-feedback velocities.
    """
    F = viscous_gas_flux_edges(Sigma_g, grid, nu, inner_boundary=inner_boundary)
    edge_sigma = interp_edges(Sigma_g)
    ve = F / np.maximum(edge_sigma, 1.0e-300)
    return 0.5 * (ve[:-1] + ve[1:])


def build_fixed_disk(grid: Dict[str, np.ndarray], params: Dict[str, Any]) -> Dict[str, np.ndarray]:
    r = grid["r"]
    area = grid["area"]
    gas = params["gas"]

    shape = gas_shape(r, params)

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
    nu_background = float(gas["alpha"]) * c_s * H
    gap_factor, gap_width_cm = gaussian_viscosity_gap(grid, params, H)
    nu = nu_background * gap_factor
    alpha_visc = float(gas["alpha"]) * gap_factor

    # Optional initialized gap: quasi-steady Sigma~1/nu *relative to the
    # original profile*, renormalized to preserve the requested total gas mass.
    # This avoids introducing a large artificial t=0 gap-clearing transient.
    # It is approximate, as the base tapered disk is not exactly steady.
    gap_cfg = gas.get("gap", {})
    if gap_cfg.get("enabled", False) and gap_cfg.get("initial_profile", "inverse_viscosity") == "inverse_viscosity":
        shape = shape / gap_factor
    m_target = float(gas["M_gas_msun"]) * MSUN
    sigma_norm = m_target / np.sum(area * shape)
    Sigma_g = sigma_norm * shape

    if gas["velocity_mode"] == "viscous":
        if gas.get("update_gas", True) and gas.get("evolution_mode", "viscous_diffusion") == "viscous_diffusion":
            v_g = gas_viscous_velocity_from_flux(
                Sigma_g, grid, nu, gas.get("viscous_inner_boundary", "outflow"))
        else:
            v_g = -1.5 * nu / r
    elif gas["velocity_mode"] == "zero":
        v_g = np.zeros_like(r)
    else:
        v_g = np.full_like(r, float(gas["constant_v_g_cm_s"]))

    rho_mid = Sigma_g / (np.sqrt(2.0 * PI) * H)
    P_mid = rho_mid * c_s**2
    dlnP_dlnr = np.gradient(np.log(P_mid), np.log(r), edge_order=2)
    eta = -0.5 * (c_s / v_K) ** 2 * dlnP_dlnr
    # The viscosity enhancement is NOT automatically a change to turbulent
    # particle diffusion: the two can represent different physical processes.
    if gap_cfg.get("enabled", False) and gap_cfg.get("diffusion_follows_viscosity", False):
        D_g = nu / float(gas["schmidt_gas"])
    else:
        D_g = nu_background / float(gas["schmidt_gas"])

    return {
        "Sigma_g": Sigma_g,
        "nu_background": nu_background,
        "gap_viscosity_factor": gap_factor,
        "gap_width_au": gap_width_cm / AU,
        "alpha_visc": alpha_visc,
        "T_mid": T_mid,
        "Omega": Omega,
        "v_K": v_K,
        "c_s": c_s,
        "H": H,
        "nu": nu,
        "v_g": v_g,
        "v_g_visc": v_g.copy(),
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
    diagnostics: Optional[Dict[str, np.ndarray]] = None,
    force_excess_destination: Optional[str] = None,
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

    def _record(key: str, value: np.ndarray) -> None:
        if diagnostics is None:
            return
        if key not in diagnostics:
            diagnostics[key] = np.zeros_like(value, dtype=float)
        diagnostics[key] += np.maximum(np.asarray(value, dtype=float), 0.0)

    def _capacity_channel(field: str) -> str:
        if field.startswith("CO2_at_H2O"):
            return "CO2_at_H2O"
        if field.startswith("CO_at_CO2"):
            return "CO_at_CO2"
        if field.startswith("CO_at_H2O"):
            return "CO_at_H2O"
        raise ValueError(f"Unknown capacity-limited guest field {field!r}.")

    if not bool(cfg.get("enabled", False)):
        return targets

    excess_destination = (
        str(force_excess_destination)
        if force_excess_destination is not None
        else str(cfg.get("excess_destination", "gas"))
    )
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
            channel = _capacity_channel(source_field)
            _record(f"dSigma_capacity_to_gas_{species}", excess)
            _record(f"dSigma_capacity_to_gas_{channel}", excess)
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
            channel = _capacity_channel(source_field)
            _record(f"dSigma_capacity_to_gas_{species}", remaining)
            _record(f"dSigma_capacity_to_gas_{channel}", remaining)

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

        channel = _capacity_channel(guest_field)
        _record(f"dSigma_capacity_excess_{channel}", excess)
        _record(f"dSigma_capacity_excess_{channel}_{carrier}", excess)

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

                _record("dSigma_capacity_excess_CO_at_H2O", co_excess)
                _record(f"dSigma_capacity_excess_CO_at_H2O_{carrier}", co_excess)
                _record("dSigma_capacity_excess_CO2_at_H2O", co2_excess)
                _record(f"dSigma_capacity_excess_CO2_at_H2O_{carrier}", co2_excess)

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



def condensation_carrier_weights(
    state: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Return radial carrier weights used to allocate shared vapor.

    ``dust.volatile_carrier_fractions`` describes the *initial* volatile
    inventory assigned to each carrier. It should not, in general, also be
    interpreted as the fraction of newly condensing vapor that lands on each
    carrier.

    In the ``surface_area`` or ``available_surface_area`` modes, raw weights
    trace the geometric grain surface/collision area of refractory cores,

        A_c propto Sigma_ref,c / (rho_s,c a_c).

    For monodisperse spherical grains, both total grain surface area and
    collisional cross section differ from this expression only by constants
    that cancel when the carrier weights are normalized.

    Refractory surface density is used rather than volatile-ice mass because it
    traces the number of grain cores. Adding an ice mantle changes a grain's
    mass but does not create additional grains in this two-carrier model.

    Survival is NOT included in these raw weights. In available_surface_area
    mode it is folded into the site scores with the reservoir fraction, then
    those scores are normalized ACROSS all physically cold eligible sites.
    The deposited gas is not multiplied by survival again. The old
    surface_area/fixed closures retain their original survival-on-gas step.

    ``fixed`` mode reproduces the legacy *carrier weighting* in which shared
    vapor is split using ``dust.volatile_carrier_fractions``. This does not
    reproduce the old model bit-for-bit because phase pooling and matrix
    history can be different. Compare with the original script for baseline.

    """
    phase_cfg = params["dust"]["phase_partition"]
    mode = str(
        phase_cfg.get("condensation_weighting", "available_surface_area")
    ).lower()

    if mode == "fixed":
        cf = params["dust"]["volatile_carrier_fractions"]
        total = sum(float(cf[c]) for c in CARRIERS)
        if total <= 0.0:
            raise ValueError(
                "dust.volatile_carrier_fractions must have positive sum when "
                "condensation_weighting='fixed'."
            )

        template = next(iter(state.values()))
        return {
            c: np.full_like(
                template,
                float(cf[c]) / total,
                dtype=float,
            )
            for c in CARRIERS
        }

    if mode not in {"surface_area", "available_surface_area"}:
        raise ValueError(
            "Unknown dust.phase_partition.condensation_weighting="
            f"{mode!r}. Use available_surface_area, surface_area or fixed."
        )

    area_proxy: Dict[str, np.ndarray] = {}
    for carrier in CARRIERS:
        cfg = params["dust"]["carriers"][carrier]
        a = float(cfg["grain_size_cm"])
        rho_s = float(cfg["rho_s_g_cm3"])

        if a <= 0.0 or rho_s <= 0.0:
            raise ValueError(
                f"Positive grain_size_cm and rho_s_g_cm3 are required for "
                f"surface-area condensation weighting ({carrier})."
            )

        sigma_ref = np.maximum(state[ref_field(carrier)], 0.0)
        area_proxy[carrier] = sigma_ref / (rho_s * a)

    total_area = np.zeros_like(
        next(iter(state.values())),
        dtype=float,
    )
    for carrier in CARRIERS:
        total_area += area_proxy[carrier]

    floor = float(
        phase_cfg.get("condensation_area_floor", 1.0e-300)
    )

    weights: Dict[str, np.ndarray] = {}
    ok = total_area > floor

    for carrier in CARRIERS:
        w = np.zeros_like(total_area)
        w[ok] = area_proxy[carrier][ok] / total_area[ok]
        weights[carrier] = w

    return weights


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
    for carrier in epsilon_by_carrier:
        disk[f"epsilon_{carrier}"] = epsilon_by_carrier[carrier]

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

    # Midplane thermal stability is a distinct quantity from vertically
    # averaged ice survival. The latter contains upper-layer exposure and must
    # not be multiplied into an inherited ice inventory at every timestep.
    midplane_thermal_survival: Dict[str, np.ndarray] = {
        "CO_pure": smooth_ice_fraction(T_mid, vol["CO"]["release_temperatures_K"]["pure"], vol["CO"]["transition_widths_K"]["pure"]),
        "CO_at_CO2": smooth_ice_fraction(T_mid, vol["CO"]["release_temperatures_K"]["at_CO2"], vol["CO"]["transition_widths_K"]["at_CO2"]),
        "CO_at_H2O": smooth_ice_fraction(T_mid, vol["CO"]["release_temperatures_K"]["at_H2O"], vol["CO"]["transition_widths_K"]["at_H2O"]),
        "CO2_pure": smooth_ice_fraction(T_mid, vol["CO2"]["release_temperatures_K"]["pure"], vol["CO2"]["transition_widths_K"]["pure"]),
        "CO2_at_H2O": smooth_ice_fraction(T_mid, vol["CO2"]["release_temperatures_K"]["at_H2O"], vol["CO2"]["transition_widths_K"]["at_H2O"]),
        "H2O": smooth_ice_fraction(T_mid, vol["H2O"]["release_temperature_K"], vol["H2O"]["transition_width_K"]),
    }

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
        "midplane_thermal_survival": midplane_thermal_survival,
        "Omega": disk["Omega"],
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
    
def compute_release(state_before, state_after, dt, grid):
    """Return additive release/source bookkeeping for one phase substep.

    ``dM_<reservoir>`` is the gross positive mass loss from that solid
    reservoir.  It can exceed the net gas source when material is reassigned
    among solid reservoirs.  Gas fields include net, positive-gain, and
    positive-loss terms so those effects can be distinguished explicitly.
    """
    area = grid["area"]
    out: Dict[str, np.ndarray] = {}

    for channel, fields in ICE_RELEASE_CHANNEL_FIELDS.items():
        dSigma_release = np.zeros_like(grid["r"])
        for field in fields:
            dSigma_release += np.maximum(state_before[field] - state_after[field], 0.0)
        out[f"dSigma_{channel}"] = dSigma_release
        out[f"dM_{channel}"] = area * dSigma_release

    for species, gas_field in PHASE_GAS_FIELDS.items():
        delta = state_after[gas_field] - state_before[gas_field]
        gain = np.maximum(delta, 0.0)
        loss = np.maximum(-delta, 0.0)
        out[f"dSigma_{species}_gas"] = delta
        out[f"dM_{species}_gas"] = area * delta
        out[f"dSigma_{species}_gas_gain"] = gain
        out[f"dM_{species}_gas_gain"] = area * gain
        out[f"dSigma_{species}_gas_loss"] = loss
        out[f"dM_{species}_gas_loss"] = area * loss

    return out


def zero_release_integral(grid: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """Create a zero-valued container for additive release diagnostics."""
    zero = np.zeros_like(grid["r"], dtype=float)
    out: Dict[str, np.ndarray] = {}
    for channel in ICE_RELEASE_CHANNEL_FIELDS:
        out[f"dSigma_{channel}"] = zero.copy()
        out[f"dM_{channel}"] = zero.copy()
    for species in PHASE_GAS_FIELDS:
        for suffix in ("", "_gain", "_loss"):
            out[f"dSigma_{species}_gas{suffix}"] = zero.copy()
            out[f"dM_{species}_gas{suffix}"] = zero.copy()
    return out


def accumulate_release(
    accumulator: Dict[str, np.ndarray],
    increment: Dict[str, np.ndarray],
) -> None:
    """Add one phase-substep diagnostic into an interval or lifetime total."""
    for key, value in increment.items():
        if key not in accumulator:
            accumulator[key] = np.zeros_like(value, dtype=float)
        accumulator[key] += value


def release_output_fields(
    interval: Dict[str, np.ndarray],
    cumulative: Dict[str, np.ndarray],
    interval_dt: float,
    grid: Dict[str, np.ndarray],
) -> Dict[str, np.ndarray]:
    """Build snapshot columns from interval and lifetime release integrals.

    Backward-compatible ``dM_*`` fields now mean the integral since the
    previous saved snapshot.  ``cum_dM_*`` fields are the direct lifetime
    integral and should be preferred by analysis scripts.
    """
    out = {key: value.copy() for key, value in interval.items()}
    for key, value in cumulative.items():
        out[f"cum_{key}"] = value.copy()

    dlnr = np.log(grid["r_edge_au"][1:] / grid["r_edge_au"][:-1])
    safe_dt = max(float(interval_dt), 1.0e-300)
    for key, value in interval.items():
        if not key.startswith("dM_"):
            continue
        suffix = key[3:]
        rate = value / safe_dt if interval_dt > 0.0 else np.zeros_like(value)
        out[f"Mdot_{suffix}"] = rate
        out[f"Mdot_dlnr_{suffix}"] = rate / dlnr

    out["release_interval_yr"] = np.full_like(grid["r"], interval_dt / YR)
    out["release_semantics_version"] = np.full_like(grid["r"], 3.0)
    return out


def capacity_output_fields(
    initial: Dict[str, np.ndarray], cumulative: Dict[str, np.ndarray],
    grid: Dict[str, np.ndarray], current: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Build rejection diagnostics for PROPOSED phase targets, not net loss.

    Saturated hosts may reject the same proposed condensation repeatedly.
    Consequently these cumulatives can greatly exceed the physical inventory;
    they must never be reported as a net volatile loss or integrated delivery.
    """
    out: Dict[str, np.ndarray] = {}
    groups = [("initial", initial), ("cum", cumulative)]
    if current is not None:
        groups.append(("current", current))
    for prefix, values in groups:
        for key, arr in values.items():
            if not key.startswith("dSigma_"):
                continue
            arr = np.asarray(arr, dtype=float)
            out[f"{prefix}_{key}"] = arr.copy()
            out[f"{prefix}_dM_{key[7:]}"] = grid["area"] * arr
    return out


def accumulate_scalar_diagnostics(total: Dict[str, float], increment: Dict[str, float]) -> None:
    """Accumulate transport/numerical scalar diagnostics."""
    for key, value in increment.items():
        value = float(value)
        if key.startswith("max_negative_"):
            total[key] = max(total.get(key, 0.0), value)
        else:
            total[key] = total.get(key, 0.0) + value


def accumulate_array_diagnostics(total: Dict[str, np.ndarray], increment: Dict[str, np.ndarray]) -> None:
    for key, value in increment.items():
        arr = np.asarray(value, dtype=float)
        if key not in total:
            total[key] = np.zeros_like(arr)
        total[key] += arr


# -----------------------------
# Initialization and phase targets
# -----------------------------
def initialize_state(
    grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
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
        state = set_phase_equilibrium(state, vertical, params, capacity_diagnostics=capacity_diagnostics)

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
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
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

    targets = apply_trapping_capacity_limits(targets, params, diagnostics=capacity_diagnostics)
    
    return targets

def _cold_condensation_allocations(
    vapor_pool: np.ndarray,
    specs: list[tuple[str, str, str, float]],
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    max_new_ice: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Newly condensed vapor by field; keep capture rate separate from site choice.

    Site scores are base grain area * eligible reservoir fraction * carrier
    mean survival. In available_surface_area mode they are normalized ACROSS
    ALL eligible carriers and matrices for this molecular species. The gas
    contribution is NOT multiplied by survival again in the phase target.

    Separately, max_survival caps the total condensed fraction by the largest
    carrier's reservoir-fraction-weighted survival. This is a fast vertical-
    mixing, rapid-capture assumption, not an adsorption/desorption kinetic law.
    The area_weighted_survival alternative is more conservative and (largely)
    reverts the previous surface-area * survival gas uptake efficiency.
    """
    pcfg = params["dust"]["phase_partition"]
    mode = str(pcfg.get("condensation_weighting", "available_surface_area")).lower()
    base = condensation_carrier_weights(state, params)
    survival_floor = float(pcfg.get("condensation_survival_floor", 1e-10))
    scores = {}
    per_carrier = {c: np.zeros_like(vapor_pool) for c in CARRIERS}
    remaining_capacity = {}
    for c, field, key, frac in specs:
        survival = np.clip(vertical["mean_survival"][c][key], 0, 1)
        survival = np.where(survival >= survival_floor, survival, 0.0)
        limit = (np.asarray(max_new_ice[field], dtype=float)
                 if max_new_ice is not None and field in max_new_ice
                 else np.full_like(vapor_pool, np.inf))
        remaining_capacity[field] = np.maximum(limit, 0)
        eligible = remaining_capacity[field] > 0
        scores[field] = np.where(eligible, base[c] * float(frac) * survival, 0.0)
        # A carrier with no physical grains must not set the global gas
        # capture efficiency. A nearly saturated host contributes at most
        # its remaining fractional capacity to the efficient-capture bound.
        capacity_fraction = np.minimum(
            1.0, np.divide(
                remaining_capacity[field], np.maximum(np.maximum(vapor_pool, 0.0), 1e-300),
                out=np.ones_like(vapor_pool),
                where=np.isfinite(remaining_capacity[field]),
            )
        )
        per_carrier[c] += np.where(
            eligible & (base[c] > 0),
            float(frac) * survival * capacity_fraction,
            0.0,
        )

    if mode != "available_surface_area":
        # Old behavior for controlled comparison, including the original
        # separate carrier-area normalization followed by survival.
        return {field: np.maximum(vapor_pool, 0) * score for field, score in scores.items()}

    score_sum = sum(scores.values())
    if str(pcfg.get("condensation_capture_mode", "max_survival")) == "max_survival":
        capture = np.maximum.reduce(list(per_carrier.values()))
    else:
        capture = sum(base[c] * per_carrier[c] for c in CARRIERS)
    capture = np.where(score_sum > 0, np.clip(capture, 0, 1), 0.0)
    amount = np.maximum(vapor_pool, 0) * capture
    # Water-filling: remove saturated guest reservoirs, then redistribute
    # unaccepted gas to the other still-cold surfaces. Any gas that cannot fit
    # stays in the vapor reservoir; it is not incorrectly added to a host.
    remaining = amount.copy()
    deposited = {field: np.zeros_like(amount) for field in scores}
    for _ in range(len(scores) + 1):
        active = {
            field: np.where(
                remaining_capacity[field] > np.maximum(1e-14 * amount, 1e-300),
                score, 0.0,
            )
            for field, score in scores.items()
        }
        norm = sum(active.values())
        ok = norm > float(pcfg.get("condensation_area_floor", 1e-300))
        if not np.any(ok & (remaining > 0)):
            break
        added_total = np.zeros_like(amount)
        for field, score in active.items():
            proposal = np.divide(remaining * score, norm,
                                 out=np.zeros_like(amount), where=ok)
            accepted = np.minimum(proposal, remaining_capacity[field])
            deposited[field] += accepted
            remaining_capacity[field] = np.maximum(remaining_capacity[field] - accepted, 0.0)
            added_total += accepted
        remaining = np.maximum(remaining - added_total, 0.0)
        if not np.any(added_total > 0):
            break
    return deposited


def _remaining_guest_condensation_capacity(
    species: str,
    specs: list[tuple[str, str, str, float]],
    retained: Dict[str, np.ndarray],
    targets: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Available new guest mass, given provisional host-ice phase targets.

    This pre-limits GUEST deposition only. The downstream capacity operator
    still clips existing ice when its host sublimates and enforces optional
    combined-host constraints. A zero-capacity site cannot attract an unlimited
    sequence of proposed deposits simply because its surface is cold.
    """
    tcfg = trapping_capacity_config(params)
    pcfg = params["dust"]["phase_partition"]
    if not bool(tcfg.get("enabled", False)) or not bool(pcfg.get("prelimit_condensation_by_capacity", True)):
        return {}
    caps = {}
    for c, field, _, _ in specs:
        host, max_ratio, guest, host_species = None, None, None, None
        if field == co2_at_h2o_field(c):
            host = targets[h2o_field(c)]
            max_ratio = tcfg.get("CO2_at_H2O", {}).get("max_guest_per_host_mol", None)
            guest, host_species = "CO2", "H2O"
        elif field == co_at_h2o_field(c):
            host = targets[h2o_field(c)]
            max_ratio = tcfg.get("CO_at_H2O", {}).get("max_guest_per_host_mol", None)
            guest, host_species = "CO", "H2O"
        elif field == co_at_co2_field(c):
            mode = tcfg.get("CO_at_CO2", {}).get("host_mode", "total_CO2_ice")
            host = targets[co2_pure_field(c)]
            if mode == "total_CO2_ice":
                host = host + targets[co2_at_h2o_field(c)]
            elif mode != "pure_CO2_ice":
                raise ValueError(f"Unknown CO2 host_mode: {mode}")
            max_ratio = tcfg.get("CO_at_CO2", {}).get("max_guest_per_host_mol", None)
            guest, host_species = "CO", "CO2"
        if host is None or max_ratio is None:
            continue
        # The host is calculated for the same carrier and phase-target epoch.
        maximum = float(max_ratio) * (MOLECULAR_WEIGHTS[guest] /
                                         MOLECULAR_WEIGHTS[host_species]) * np.maximum(host, 0)
        caps[field] = np.maximum(maximum - retained[field], 0.0)
    return caps


def _carrier_history_targets(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    *,
    allow_resequestration: bool,
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
    keep_existing_ice: bool = False,
) -> Dict[str, np.ndarray]:
    """Carrier and optionally matrix-history-preserving phase targets.

    Existing ice only survives in its own matrix (or sublimates). Newly
    released material joins the same shared gas pool as existing vapor, then
    may condense on eligible cold surfaces. Newly deposited guest material
    is pre-limited by host capacity, with final capacity enforcement last.

    This avoids reassigning all pre-existing ice to prescribed reservoir
    fractions at every timestep, which previously caused recurring, largely
    artificial over-capacity proposals and apparent reservoir cycling.
    """
    targets = {name: np.zeros_like(next(iter(state.values()))) for name in ALL_FIELDS}
    ms = vertical["mean_survival"]
    for c in CARRIERS:
        targets[ref_field(c)] = state[ref_field(c)].copy()
    preserve_matrix = bool(params["dust"]["phase_partition"].get("preserve_matrix_history", True))
    co_fr = params["volatiles"]["CO"]["solid_fractions"]
    co2_fr = params["volatiles"]["CO2"]["solid_fractions"]
    reservoir_sets = {
        "H2O": [(c, h2o_field(c), "H2O", 1.0) for c in CARRIERS],
        "CO2": ([(c, co2_pure_field(c), "CO2_pure", float(co2_fr["pure"])) for c in CARRIERS]
                + [(c, co2_at_h2o_field(c), "CO2_at_H2O", float(co2_fr["at_H2O"])) for c in CARRIERS]),
        "CO": ([(c, co_pure_field(c), "CO_pure", float(co_fr["pure"])) for c in CARRIERS]
               + [(c, co_at_co2_field(c), "CO_at_CO2", float(co_fr["at_CO2"])) for c in CARRIERS]
               + [(c, co_at_h2o_field(c), "CO_at_H2O", float(co_fr["at_H2O"])) for c in CARRIERS]),
    }
    for species, specs in reservoir_sets.items():
        # Every target construction conserves this molecular species locally.
        total = state[f"{species}_gas"] + sum(state[field] for _, field, _, _ in specs)
        retained = {}
        if preserve_matrix:
            for c, field, key, _ in specs:
                # During evolution, physical release has ALREADY been applied
                # using an elapsed-time exponential operator. Never remove
                # (1-mean_survival) of the remaining ice every numerical call.
                # At t=0, initialize thermal equilibrium at the MIDPLANE only;
                # upper-layer exposure has not had an arbitrary time to work.
                retained[field] = (state[field].copy() if keep_existing_ice else
                                   vertical["midplane_thermal_survival"][key] * state[field])
        else:
            # Legacy: existing ice continuously redistributes between matrices.
            for c in CARRIERS:
                old_carrier = sum(state[field] for cc, field, _, _ in specs if cc == c)
                for cc, field, key, frac in specs:
                    if cc == c:
                        retained[field] = float(frac) * ms[c][key] * old_carrier

        gas_pool = np.maximum(total - sum(retained.values()), 0.0)
        if allow_resequestration or species == "H2O":
            possible_sites = specs
        else:
            # In the irreversible case, CO/CO2 mixed matrices cannot be rebuilt.
            # Pure ice gets unit weighting rather than its mixed-ice fraction.
            possible_sites = [
                (c, field, key, 1.0) for c, field, key, _ in specs
                if field == co_pure_field(c) or field == co2_pure_field(c)
            ]
        max_new_ice = _remaining_guest_condensation_capacity(
            species, possible_sites, retained, targets, params
        )
        deposited = _cold_condensation_allocations(
            gas_pool, possible_sites, state, vertical, params,
            max_new_ice=max_new_ice,
        )
        for _, field, _, _ in specs:
            targets[field] = np.maximum(retained[field] + deposited.get(field, 0.0), 0.0)
        targets[f"{species}_gas"] = np.maximum(
            total - sum(targets[field] for _, field, _, _ in specs), 0.0
        )

    return apply_trapping_capacity_limits(
        targets, params, diagnostics=capacity_diagnostics,
        force_excess_destination=None if allow_resequestration else "gas",
    )


def phase_targets_w_history(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
    keep_existing_ice: bool = False,
) -> Dict[str, np.ndarray]:
    return _carrier_history_targets(state, vertical, params, allow_resequestration=True,
                                    capacity_diagnostics=capacity_diagnostics,
                                    keep_existing_ice=keep_existing_ice)


def phase_targets_w_history_no_resequestration(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
    keep_existing_ice: bool = False,
) -> Dict[str, np.ndarray]:
    return _carrier_history_targets(state, vertical, params, allow_resequestration=False,
                                    capacity_diagnostics=capacity_diagnostics,
                                    keep_existing_ice=keep_existing_ice)


def build_phase_targets(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
    keep_existing_ice: bool = False,
) -> Dict[str, np.ndarray]:
    """Select the configured phase-target model."""
    phase_cfg = params["dust"]["phase_partition"]

    if not bool(phase_cfg.get("allow_resequestration", True)):
        return phase_targets_w_history_no_resequestration(
            state,
            vertical,
            params,
            capacity_diagnostics=capacity_diagnostics,
            keep_existing_ice=keep_existing_ice,
        )

    if bool(phase_cfg["preserve_carrier_history"]):
        return phase_targets_w_history(
            state,
            vertical,
            params,
            capacity_diagnostics=capacity_diagnostics,
            keep_existing_ice=keep_existing_ice,
        )

    return phase_targets(
        state,
        vertical,
        params,
        capacity_diagnostics=capacity_diagnostics,
    )

def set_phase_equilibrium(
    state: Dict[str, np.ndarray],
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    capacity_diagnostics: Optional[Dict[str, np.ndarray]] = None,
) -> Dict[str, np.ndarray]:
    """Set the state to its phase target and optionally record initial cap rejection."""
    targets = build_phase_targets(
        state, vertical, params, capacity_diagnostics=capacity_diagnostics
    )
    for name in ALL_FIELDS:
        state[name] = targets[name]
    return state

def ice_loss_rate_components(
    carrier: str,
    key: str,
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return thermal and mixing/exposure loss rates [s^-1] for existing ice.

    Temperature-driven release in a warm midplane is rapid (tau_phase); slow
    processing of ice in a cold disk occurs through access to vertically
    exposed layers on t_mix = factor / (alpha_z Omega). The exposure fraction
    is set by the difference between midplane thermal survival and the
    carrier-weighted, thermally/shielding-limited survival fraction.

    This is a deliberately simple two-timescale closure, not a full vertical
    chemistry/photodesorption solver. It has a continuous-time limit and is
    independent of how often the numerical phase operator is called.
    """
    cfg = params["dust"]["phase_partition"]
    s_mid = np.clip(vertical["midplane_thermal_survival"][key], 0.0, 1.0)
    s_mean = np.clip(vertical["mean_survival"][carrier][key], 0.0, 1.0)
    tau_thermal = float(params["volatiles"]["phase_relaxation_time_yr"]) * YR
    k_thermal = (1.0 - s_mid) / tau_thermal
    k_vertical = np.zeros_like(s_mid)
    if cfg.get("ice_loss_mode", "mixing_limited") == "mixing_limited":
        alpha_z = max(0.0, float(params["vertical"]["alpha_z"]))
        efficiency = float(cfg.get("vertical_exposure_efficiency", 1.0))
        factor = float(cfg.get("vertical_mixing_timescale_factor", 1.0))
        # Count the upper-layer exposure only in the fraction that has NOT
        # already been identified as rapidly thermally unstable at midplane.
        exposed = np.clip(s_mid - s_mean, 0.0, 1.0)
        k_vertical = (efficiency * alpha_z / factor) * vertical["Omega"] * exposed
    return k_thermal, k_vertical


def apply_elapsed_time_ice_loss(
    state: Dict[str, np.ndarray],
    dt: float,
    vertical: Dict[str, Any],
    params: Dict[str, Any],
) -> Dict[str, np.ndarray]:
    """Release existing ice to its gas species with exact exponential kinetics.

    Do not apply the extra phase-relaxation interpolation to this loss: doing
    so would again make depletion depend on the numerical timestep.
    """
    for carrier in CARRIERS:
        for field in solid_fields_for(carrier):
            if field.startswith("ref_solid"):
                continue
            species = FIELD_TO_SPECIES[field]
            k_thermal, k_vertical = ice_loss_rate_components(
                carrier, FIELD_TO_SURVIVAL_KEY[field], vertical, params
            )
            fraction = -np.expm1(-dt * (k_thermal + k_vertical))
            release = np.maximum(state[field], 0.0) * np.clip(fraction, 0.0, 1.0)
            state[field] = np.maximum(state[field] - release, 0.0)
            state[f"{species}_gas"] += release
    return state


def phase_relaxation_step(
    state: Dict[str, np.ndarray],
    dt: float,
    vertical: Dict[str, Any],
    params: Dict[str, Any],
    return_diagnostics: bool = False,
):
    """Release ice on physical timescales, then relax new condensation.

    Crucially, pre-existing ice is NOT multiplied by its carrier-averaged
    survival fraction every call. A physical exponential loss is applied once
    over dt, and only deposition from the new vapor pool is phase-relaxed.
    """
    tau = float(params["volatiles"]["phase_relaxation_time_yr"]) * YR
    fac = math.exp(-dt / tau)
    relax_fraction = -math.expm1(-dt / tau)
    phase_cfg = params["dust"]["phase_partition"]

    if bool(phase_cfg.get("preserve_matrix_history", True)) and bool(phase_cfg.get("preserve_carrier_history", True)):
        # 1. Existing ice loss: continuous-time, conservative operator.
        state = apply_elapsed_time_ice_loss(state, dt, vertical, params)

        # 2. Compute eligible NEW condensation on the surviving carrier ice.
        # Using keep_existing_ice avoids the old (S_mean)**N time-step artifact.
        capacity_target_diag: Dict[str, np.ndarray] = {}
        targets = build_phase_targets(
            state, vertical, params,
            capacity_diagnostics=capacity_target_diag,
            keep_existing_ice=True,
        )
        for name in ALL_FIELDS:
            if name.startswith("ref_solid"):
                continue
            state[name] = state[name] + relax_fraction * (targets[name] - state[name])

        # 3. Capacity is a physical upper bound even at sub-thermal dt; reject
        # guests immediately when a host has disappeared, conserving molecules.
        capacity_realized: Dict[str, np.ndarray] = {
            key: relax_fraction * value for key, value in capacity_target_diag.items()
        }
        capacity_instant: Dict[str, np.ndarray] = {}
        state = apply_trapping_capacity_limits(
            state, params, diagnostics=capacity_instant,
            force_excess_destination=("gas" if not bool(phase_cfg.get("allow_resequestration", True)) else None),
        )
        for key, arr in capacity_instant.items():
            if key not in capacity_realized:
                capacity_realized[key] = np.zeros_like(arr)
            capacity_realized[key] += arr
    else:
        # Legacy redistribution, retained for the optional model that does
        # not preserve ice/carrier history. It is NOT the fiducial closure.
        capacity_target_diag = {}
        targets = build_phase_targets(
            state, vertical, params, capacity_diagnostics=capacity_target_diag
        )
        for name in ALL_FIELDS:
            if name.startswith("ref_solid"):
                continue
            state[name] = targets[name] + (state[name] - targets[name]) * fac
        capacity_realized = {
            key: relax_fraction * value for key, value in capacity_target_diag.items()
        }

    if not return_diagnostics:
        return state
    return state, capacity_realized

def check_phase_conservation(
    before: Dict[str, np.ndarray],
    after: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Dict[str, float]:
    """Verify and quantify local molecular-species conservation."""
    do_check = bool(params["numerics"].get("check_phase_conservation", True))
    rtol = float(params["numerics"].get("phase_conservation_rtol", 1.0e-10))
    atol = float(params["numerics"].get("phase_conservation_atol", 1.0e-30))
    totals = {"CO": total_CO, "CO2": total_CO2, "H2O": total_H2O}
    metrics: Dict[str, float] = {}
    for species, func in totals.items():
        b = func(before)
        a = func(after)
        err = np.abs(a - b)
        scale = np.maximum(np.maximum(np.abs(a), np.abs(b)), atol)
        rel = err / np.maximum(scale, 1.0e-300)
        metrics[f"phase_max_abs_error_{species}"] = float(np.nanmax(err))
        metrics[f"phase_max_rel_error_{species}"] = float(np.nanmax(rel))
        tol = atol + rtol * np.maximum(np.maximum(np.abs(a), np.abs(b)), 1.0e-300)
        if do_check and np.any(err > tol):
            j = int(np.nanargmax(err / np.maximum(tol, 1.0e-300)))
            raise RuntimeError(
                f"Phase update failed local {species} conservation at radial cell {j}: "
                f"before={b[j]:.8e}, after={a[j]:.8e}, "
                f"abs_error={err[j]:.3e}, tolerance={tol[j]:.3e}."
            )
    return metrics

# -----------------------------
# Finite-volume transport
# -----------------------------
def interp_edges(arr: np.ndarray) -> np.ndarray:
    edge = np.empty(arr.size + 1, dtype=arr.dtype)
    edge[0] = arr[0]
    edge[-1] = arr[-1]
    edge[1:-1] = 0.5 * (arr[:-1] + arr[1:])
    return edge

def _solve_tridiagonal(lower: np.ndarray, diag: np.ndarray,
                       upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas solver for a diagonally dominant tridiagonal M-matrix."""
    n = len(rhs)
    c = np.empty(n, dtype=float)
    d = np.empty(n, dtype=float)
    denom = diag[0]
    if not np.isfinite(denom) or denom <= 0:
        raise FloatingPointError("Invalid implicit-viscosity matrix diagonal")
    c[0] = upper[0] / denom
    d[0] = rhs[0] / denom
    for i in range(1, n):
        denom = diag[i] - lower[i] * c[i-1]
        if not np.isfinite(denom) or denom <= 0:
            raise FloatingPointError("Invalid implicit-viscosity pivot")
        c[i] = upper[i] / denom if i < n-1 else 0.0
        d[i] = (rhs[i] - lower[i] * d[i-1]) / denom
    out = np.empty(n, dtype=float)
    out[-1] = d[-1]
    for i in range(n-2, -1, -1):
        out[i] = d[i] - c[i] * out[i+1]
    return out


def propose_gas_viscous_step(
    disk: Dict[str, np.ndarray], dt: float, grid: Dict[str, np.ndarray],
    params: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Conservative backward-Euler viscous evolution plus explicit dust feedback.

    We evolve
      dSigma_g/dt = -1/r d/dr [r(F_nu + Sigma_g v_feedback)],
      F_nu = -3 A(r)/sqrt(r) d/dr (nu Sigma_g sqrt(r)),
      v_feedback = 2 B eta v_K.

    A, B, eta, nu are frozen at the *start* of the timestep. The positive
    viscous diffusion operator is solved implicitly using an M-matrix;
    pressure feedback is upwinded explicitly. The caller enforces an outflow
    CFL on the resulting total gas flux before accepting this proposal.

    Returns new Sigma_g, conservative total gas mass flux at edges, diagnostics.
    No disk arrays are modified until the caller accepts the step.
    """
    old = disk["Sigma_g"]
    r, re, area = grid["r"], grid["r_edge"], grid["area"]
    n = old.size
    A = np.asarray(disk.get("backreaction_A", np.ones_like(old)), dtype=float)
    B = np.asarray(disk.get("backreaction_B", np.zeros_like(old)), dtype=float)
    nu = disk["nu"]

    # Only the pressure term is explicitly advected. It uses the *old* state,
    # preserving outflow-only boundaries in the existing transport operator.
    v_feedback = 2.0 * B * disk["eta"] * disk["v_K"]
    rhs_feedback, flux_feedback = transport_rhs(
        old, v_feedback, disk["D_g"], old, grid, diffusion_on=False
    )
    rhs = old + dt * rhs_feedback
    if not np.all(np.isfinite(rhs)) or np.any(rhs < -1.0e-12 * np.maximum(old, 1.0e-30)):
        raise FloatingPointError("Explicit dust-feedback gas step violates positivity; reduce dt")
    rhs = np.maximum(rhs, 0.0)

    lower = np.zeros(n)
    diag = np.ones(n)
    upper = np.zeros(n)
    Ae = interp_edges(A)
    for i in range(n-1):
        e = i + 1
        coeff = 3.0 * Ae[e] / (np.sqrt(re[e]) * (r[i+1] - r[i]))
        left_factor = coeff * nu[i] * np.sqrt(r[i])
        right_factor = coeff * nu[i+1] * np.sqrt(r[i+1])
        interface = dt * 2.0 * PI * re[e]
        diag[i] += interface * left_factor / area[i]
        upper[i] -= interface * right_factor / area[i]
        lower[i+1] -= interface * left_factor / area[i+1]
        diag[i+1] += interface * right_factor / area[i+1]

    # Inner outflow / zero-torque Robin condition: F_nu,0=-k_inner Sigma_0.
    mode = params["gas"].get("viscous_inner_boundary", "outflow")
    k_inner = Ae[0] * viscous_inner_outflow_coefficient(grid, nu, mode)
    diag[0] += dt * 2.0 * PI * re[0] * k_inner / area[0]

    new = _solve_tridiagonal(lower, diag, upper, rhs)
    if not np.all(np.isfinite(new)) or np.any(new < -1.0e-11 * np.maximum(old, 1.0e-30)):
        raise FloatingPointError("Implicit viscous gas step violated positivity")
    new = np.maximum(new, 0.0)
    flux_viscous = viscous_gas_flux_edges(
        new, grid, nu, A=A, inner_boundary=mode)
    flux_total = flux_viscous + flux_feedback

    # Mass conservation must hold to floating-point precision, even at a
    # moving feedback-generated front. The floor is accounted for separately.
    expected = old - dt * (2.0 * PI / area) * (
        re[1:] * flux_total[1:] - re[:-1] * flux_total[:-1]
    )
    err = float(np.max(np.abs(expected-new) / np.maximum(1.0, np.abs(new))))
    if err > 1.0e-9:
        raise FloatingPointError(f"Gas viscous flux/solution inconsistency: {err:.3e}")

    floor = float(params["gas"].get("Sigma_floor", 0.0))
    floor_add = np.maximum(floor-new, 0.0)
    clipped = np.maximum(new, floor)
    diagnostics = {
        "boundary_inner_gas_disk_g": max(-2.0 * PI * re[0] * float(flux_total[0]), 0.0) * dt,
        "boundary_outer_gas_disk_g": max(2.0 * PI * re[-1] * float(flux_total[-1]), 0.0) * dt,
        "clipped_added_gas_disk_g": float(np.sum(area * floor_add)),
        "max_negative_gas_disk": max(-float(np.min(new)), 0.0),
        "clip_events_gas_disk": float(np.count_nonzero(new < floor)),
    }
    return clipped, flux_total, diagnostics


def gas_flux_cfl_time(
    Sigma_g: np.ndarray, flux: np.ndarray,
    grid: Dict[str, np.ndarray], cfl: float,
) -> float:
    """Maximum timestep for an explicitly transported passive gas tracer.

    Unlike a velocity-only CFL, this also resolves large diffusion-driven
    mass fluxes generated by a steep Sigma_g gradient.
    """
    re, area = grid["r_edge"], grid["area"]
    outflow = 2.0 * PI * (
        re[1:] * np.maximum(flux[1:], 0.0)
        + re[:-1] * np.maximum(-flux[:-1], 0.0)
    )
    available = area * np.maximum(Sigma_g, 0.0)
    good = (outflow > 0.0) & (available > 0.0)
    if not np.any(good):
        return np.inf
    return float(cfl) * float(np.min(available[good] / outflow[good]))


def gas_tracer_transport_rhs(
    Sigma: np.ndarray,
    gas_flux_edges: np.ndarray,
    D: np.ndarray,
    Sigma_g: np.ndarray,
    grid: Dict[str, np.ndarray],
    diffusion_on: bool,
) -> Tuple[np.ndarray, np.ndarray]:
    """Conservative gas-tracer transport using the actual gas mass flux.

    This guarantees that a spatially constant vapor/gas mixing ratio remains
    constant (without a phase source), even during rapid viscous evolution.
    """
    ratio = Sigma / np.maximum(Sigma_g, 1.0e-300)
    left = np.zeros(len(Sigma) + 1)
    right = np.zeros(len(Sigma) + 1)
    left[1:] = ratio
    right[:-1] = ratio
    upwind = np.where(gas_flux_edges >= 0.0, left, right)
    flux = gas_flux_edges * upwind
    if diffusion_on:
        r = grid["r"]
        D_e = interp_edges(D)
        gas_e = interp_edges(Sigma_g)
        derivative = np.zeros(len(Sigma)+1)
        derivative[1:-1] = (ratio[1:]-ratio[:-1])/(r[1:]-r[:-1])
        flux -= D_e * gas_e * derivative
    re, area = grid["r_edge"], grid["area"]
    rhs = -2.0 * PI / area * (re[1:] * flux[1:] - re[:-1] * flux[:-1])
    return rhs, flux


def update_gas_surface_density_prescribed(
    disk: Dict[str, np.ndarray], dt: float, grid: Dict[str, np.ndarray],
    params: Dict[str, Any], return_diagnostics: bool = False,
):
    Sigma_old = disk["Sigma_g"]
    rhs, flux = transport_rhs(Sigma_old, disk["v_g"], disk["D_g"], Sigma_old, grid, diffusion_on=False)
    raw = Sigma_old + dt * rhs
    floor = float(params["gas"].get("Sigma_floor", 1.0e-30))
    new = np.maximum(raw, floor)
    disk["Sigma_g"] = new
    if not return_diagnostics:
        return disk
    inner = max(-2.0 * PI * grid["r_edge"][0] * float(flux[0]), 0.0) * dt
    outer = max(2.0 * PI * grid["r_edge"][-1] * float(flux[-1]), 0.0) * dt
    correction = np.maximum(new - raw, 0.0)
    return disk, {
        "boundary_inner_gas_disk_g": inner,
        "boundary_outer_gas_disk_g": outer,
        "clipped_added_gas_disk_g": float(np.sum(grid["area"] * correction)),
        "max_negative_gas_disk": max(-float(np.nanmin(raw)), 0.0),
        "clip_events_gas_disk": float(np.count_nonzero(raw < floor)),
    }

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
        if gas.get("update_gas", True) and gas.get("evolution_mode", "viscous_diffusion") == "viscous_diffusion":
            v_g = gas_viscous_velocity_from_flux(
                Sigma_g, grid, nu, gas.get("viscous_inner_boundary", "outflow"))
        else:
            v_g = -1.5 * nu / r
    elif gas["velocity_mode"] == "zero":
        v_g = np.zeros_like(r)
    elif gas["velocity_mode"] == "constant":
        v_g = np.full_like(r, float(gas["constant_v_g_cm_s"]))
    else:
        raise ValueError("Unknown velocity_mode")

    if gas.get("gap", {}).get("enabled", False) and gas["gap"].get("diffusion_follows_viscosity", False):
        D_g = nu / float(gas["schmidt_gas"])
    else:
        D_g = disk["nu_background"] / float(gas["schmidt_gas"])

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
    grid: Dict[str, np.ndarray], disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]], params: Dict[str, Any],
    return_diagnostics: bool = False,
):
    num = params["numerics"]
    dr = grid["dr"]
    vmax = np.abs(disk["v_g"]).copy()
    for carrier in CARRIERS:
        vmax = np.maximum(vmax, np.abs(carrier_coeff[carrier]["v"]))
    # For full viscous gas evolution, A*v_visc and v_feedback may nearly
    # cancel; the explicit pressure-feedback term needs its own CFL bound.
    if (bool(params["gas"].get("update_gas", False))
        and params["gas"].get("evolution_mode", "viscous_diffusion") == "viscous_diffusion"):
        B = disk.get("backreaction_B", np.zeros_like(vmax))
        v_feedback = 2.0 * B * disk["eta"] * disk["v_K"]
        vmax = np.maximum(vmax, np.abs(v_feedback))
    mask_v = vmax > 0.0
    dt_adv = float(num["cfl_advective"]) * float(np.min(dr[mask_v] / vmax[mask_v])) if np.any(mask_v) else np.inf
    Dmax = (disk["D_g"].copy() if bool(params["volatiles"]["include_vapor_diffusion"])
            else np.zeros_like(disk["D_g"]))
    for carrier in CARRIERS:
        if bool(params["dust"]["carriers"][carrier]["include_diffusion"]):
            Dmax = np.maximum(Dmax, carrier_coeff[carrier]["D"])
    mask_D = Dmax > 0.0
    dt_diff = float(num["cfl_diffusive"]) * float(np.min(dr[mask_D] ** 2 / Dmax[mask_D])) if np.any(mask_D) else np.inf
    dt_max = float(num["max_timestep_yr"]) * YR
    candidates = {"advective": dt_adv, "diffusive": dt_diff, "max_timestep": dt_max}
    limiter = min(candidates, key=candidates.get)
    dt = candidates[limiter]
    if not np.isfinite(dt) or dt <= 0:
        raise RuntimeError("Computed invalid timestep.")
    if return_diagnostics:
        return dt, {"limiter": limiter, "dt_advective_s": dt_adv, "dt_diffusive_s": dt_diff, "dt_max_s": dt_max}
    return dt

def transport_step(
    state: Dict[str, np.ndarray], dt: float, grid: Dict[str, np.ndarray],
    disk: Dict[str, np.ndarray], carrier_coeff: Dict[str, Dict[str, np.ndarray]],
    params: Dict[str, Any], return_diagnostics: bool = False,
    gas_flux_edges: Optional[np.ndarray] = None,
):
    Sigma_g = disk["Sigma_g"]
    new_state = {k: v.copy() for k, v in state.items()}
    diagnostics: Dict[str, float] = {}

    def add(key: str, value: float) -> None:
        diagnostics[key] = diagnostics.get(key, 0.0) + float(value)

    def record_flux(field: str, flux: np.ndarray) -> None:
        inner = max(-2.0 * PI * grid["r_edge"][0] * float(flux[0]), 0.0) * dt
        outer = max(2.0 * PI * grid["r_edge"][-1] * float(flux[-1]), 0.0) * dt
        species = FIELD_TO_SPECIES[field]
        add(f"boundary_inner_{species}_g", inner)
        add(f"boundary_outer_{species}_g", outer)
        if field in SOLID_FIELDS:
            carrier = FIELD_TO_CARRIER[field]
            add(f"boundary_inner_{carrier}_solids_g", inner)
            add(f"boundary_outer_{carrier}_solids_g", outer)

    vapor_diffusion = bool(params["volatiles"]["include_vapor_diffusion"])
    for name in VAPOR_FIELDS:
        if gas_flux_edges is None:
            rhs, flux = transport_rhs(state[name], disk["v_g"], disk["D_g"], Sigma_g, grid, diffusion_on=vapor_diffusion)
        else:
            rhs, flux = gas_tracer_transport_rhs(state[name], gas_flux_edges, disk["D_g"], Sigma_g, grid, diffusion_on=vapor_diffusion)
        new_state[name] = state[name] + dt * rhs
        record_flux(name, flux)
    for carrier in CARRIERS:
        diffusion_on = bool(params["dust"]["carriers"][carrier]["include_diffusion"])
        for name in solid_fields_for(carrier):
            rhs, flux = transport_rhs(state[name], carrier_coeff[carrier]["v"], carrier_coeff[carrier]["D"], Sigma_g, grid, diffusion_on=diffusion_on)
            new_state[name] = state[name] + dt * rhs
            record_flux(name, flux)

    if bool(params["numerics"]["clip_negative"]):
        floor = float(params["numerics"]["surface_density_floor"])
        tol = float(params["numerics"]["negative_tolerance"])
        for name in ALL_FIELDS:
            raw = new_state[name]
            minval = float(np.nanmin(raw))
            threshold = max(tol, 1.0e-12 * max(1.0, float(np.nanmax(np.abs(raw)))))
            if minval < -threshold:
                print(f"WARNING: clipping negative values in {name}; min={minval:.3e}. Consider reducing CFL or max_timestep.", file=sys.stderr)
            clipped = np.maximum(raw, floor)
            correction = np.maximum(clipped - raw, 0.0)
            species = FIELD_TO_SPECIES[name]
            add(f"clipped_added_{species}_g", float(np.sum(grid["area"] * correction)))
            add(f"clip_events_{species}", float(np.count_nonzero(raw < floor)))
            diagnostics[f"max_negative_{species}"] = max(diagnostics.get(f"max_negative_{species}", 0.0), max(-minval, 0.0))
            new_state[name] = clipped
    return (new_state, diagnostics) if return_diagnostics else new_state

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
        # Absolute volatile elemental columns in mol cm^-2 (not normalized
        # ratios). These distinguish carbon addition from oxygen depletion.
        "C_mol_cm2_pebble": C_peb,
        "O_mol_cm2_pebble": O_peb,
        "C_mol_cm2_small": C_small,
        "O_mol_cm2_small": O_small,
        "C_mol_cm2_gas": C_gas,
        "O_mol_cm2_gas": O_gas,
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
        # Explicit aliases clarify that "hidden" here means assigned to a
        # mixed/matrix-associated reservoir, not necessarily unobservable.
        "matrix_associated_CO_fraction": (
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
    params: Optional[Dict[str, Any]] = None,
) -> None:
    co = carbon_oxygen_columns(state)
    cols: Dict[str, np.ndarray] = {
        "time_yr": np.full_like(grid["r_au"], t / YR),
        "r_au": grid["r_au"],
        "Sigma_g": disk["Sigma_g"],
        "T_mid_K": disk["T_mid"],
        "H_over_r": disk["H"] / grid["r"],
        "v_g_cm_s": disk["v_g"],
        "v_g_visc_cm_s": disk["v_g_visc"],
        "eta": disk["eta"],
        "D_g_cm2_s": disk["D_g"],
        "nu_visc_cm2_s": disk["nu"],
        "alpha_visc": disk["alpha_visc"],
        "gap_viscosity_factor": disk["gap_viscosity_factor"],
    }
    
    if "backreaction_A" in disk:
        cols["backreaction_A"] = disk["backreaction_A"]
        cols["backreaction_B"] = disk["backreaction_B"]
        cols["v_g_phi_subkep"] = disk["v_g_phi_subkep"]

    for carrier in CARRIERS:
        cols[f"St_{carrier}"] = carrier_coeff[carrier]["St"]
        cols[f"v_{carrier}_cm_s"] = carrier_coeff[carrier]["v"]
        cols[f"D_{carrier}_cm2_s"] = carrier_coeff[carrier]["D"]
        cols[f"Hd_over_H_{carrier}"] = carrier_coeff[carrier]["H_ratio"]
        if f"epsilon_{carrier}" in disk:
            cols[f"epsilon_{carrier}"] = disk[f"epsilon_{carrier}"]

    for key in SURVIVAL_KEYS:
        cols[f"snow_surface_z_over_r_{key}"] = vertical["snow_surfaces_z_over_r"][key]
        for carrier in CARRIERS:
            cols[f"mean_survival_{key}_{carrier}"] = vertical["mean_survival"][carrier][key]
            if params is not None:
                k_th, k_ex = ice_loss_rate_components(carrier, key, vertical, params)
                cols[f"thermal_ice_loss_rate_yr_{key}_{carrier}"] = k_th * YR
                cols[f"vertical_ice_loss_rate_yr_{key}_{carrier}"] = k_ex * YR
        cols[f"midplane_thermal_survival_{key}"] = vertical["midplane_thermal_survival"][key]

    # Diagnostic proposal probabilities: fraction of a unit shared gas pool
    # targeting a given carrier BEFORE host-capacity rejection. They document
    # where the cold available area actually sends vapor in the new closure.
    if params is not None:
        base = condensation_carrier_weights(state, params)
        for c in CARRIERS:
            cols[f"geometric_area_weight_{c}"] = base[c]
        cp = params["volatiles"]["CO"]["solid_fractions"]
        c2p = params["volatiles"]["CO2"]["solid_fractions"]
        spec_map = {
            "H2O": [(c, h2o_field(c), "H2O", 1.0) for c in CARRIERS],
            "CO2": [(c, co2_pure_field(c), "CO2_pure", float(c2p["pure"])) for c in CARRIERS]
                   + [(c, co2_at_h2o_field(c), "CO2_at_H2O", float(c2p["at_H2O"])) for c in CARRIERS],
            "CO": [(c, co_pure_field(c), "CO_pure", float(cp["pure"])) for c in CARRIERS]
                  + [(c, co_at_co2_field(c), "CO_at_CO2", float(cp["at_CO2"])) for c in CARRIERS]
                  + [(c, co_at_h2o_field(c), "CO_at_H2O", float(cp["at_H2O"])) for c in CARRIERS],
        }
        reseq = bool(params["dust"]["phase_partition"].get("allow_resequestration", True))
        for species, specs in spec_map.items():
            if species != "H2O" and not reseq:
                specs = [(c, field, key, 1.0) for c, field, key, _ in specs
                         if field == co_pure_field(c) or field == co2_pure_field(c)]
            probs = _cold_condensation_allocations(
                np.ones_like(grid["r"]), specs, state, vertical, params
            )
            cols[f"proposed_gas_capture_fraction_{species}"] = sum(probs.values())
            for c in CARRIERS:
                cols[f"proposed_gas_deposition_{species}_{c}"] = sum(
                    prob for field, prob in probs.items() if field.endswith("_" + c)
                )

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
        
    # Bulk H/He gas and volatile-vapor surface-density bins.
    w_gas = vertical["weights"]["gas"]

    # Save the normalized vertical gas weights.
    data["weight_gas"] = w_gas

    # Bulk H/He gas surface density in each r-z bin.
    # Summing over the vertical axis recovers disk["Sigma_g"].
    data["surfbin_gas_bulk"] = distribute_surface_density_rz(
        disk["Sigma_g"],
        w_gas,
    )

    # Gas vapor surface-density bins.
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


def diagnostics_fieldnames() -> list[str]:
    base = [
        "snapshot", "step", "time_yr", "dt_yr", "M_gas_msun", "M_gas_fixed_msun",
        "M_ref_pebble_mearth", "M_ref_small_mearth",
        "M_CO_total_mearth", "M_CO_gas_mearth", "M_CO_solid_mearth", "M_CO_hidden_mearth", "M_CO_matrix_mearth", "M_CO_pebble_mearth", "M_CO_small_mearth",
        "M_CO2_total_mearth", "M_CO2_gas_mearth", "M_CO2_solid_mearth", "M_CO2_pebble_mearth", "M_CO2_small_mearth",
        "M_H2O_total_mearth", "M_H2O_gas_mearth", "M_H2O_solid_mearth", "M_H2O_pebble_mearth", "M_H2O_small_mearth",
        "retained_CO_fraction", "retained_CO2_fraction", "retained_H2O_fraction",
        "min_sigma", "max_sigma",
        "max_epsilon_pebble", "median_epsilon_pebble", "mass_weighted_epsilon_pebble",
        "max_backreaction_X", "median_backreaction_X", "max_backreaction_Y", "median_backreaction_Y",
        "max_backreaction_A", "median_backreaction_A", "max_backreaction_B", "median_backreaction_B",
        "min_vr_g", "max_vr_g", "min_pebble_v", "max_pebble_v",
        "max_rel_delta_v_gas_backreaction", "mass_weighted_rel_delta_v_gas_backreaction",
        "max_rel_delta_v_pebble_backreaction", "mass_weighted_rel_delta_v_pebble_backreaction",
    ]
    for target in ("CO", "CO2", "H2O", "pebble_solids", "small_solids", "gas_disk"):
        base += [f"cum_boundary_inner_{target}_mearth", f"cum_boundary_outer_{target}_mearth"]
    for species in ("CO", "CO2", "H2O", "refractory", "gas_disk"):
        base += [f"cum_clipped_added_{species}_mearth", f"max_negative_{species}", f"clip_events_{species}"]
    for species in ("CO", "CO2", "H2O"):
        base += [
            f"cum_phase_gas_gain_{species}_mearth", f"cum_phase_gas_loss_{species}_mearth", f"cum_phase_gas_net_{species}_mearth",
            f"phase_cycling_factor_{species}", f"phase_max_abs_error_{species}", f"phase_max_rel_error_{species}",
            f"mass_balance_residual_{species}_mearth", f"mass_balance_residual_fraction_{species}",
        ]
    for channel in ("CO_at_CO2", "CO_at_H2O", "CO2_at_H2O"):
        base += [
            f"initial_capacity_excess_{channel}_mearth", f"current_capacity_excess_{channel}_mearth",
            f"current_capacity_active_cell_fraction_{channel}", f"cum_capacity_excess_{channel}_mearth",
            f"total_capacity_excess_{channel}_mearth",
        ]
    for species in ("CO", "CO2"):
        base += [
            f"initial_capacity_to_gas_{species}_mearth", f"current_capacity_to_gas_{species}_mearth",
            f"cum_capacity_to_gas_{species}_mearth", f"total_capacity_to_gas_{species}_mearth",
        ]
    base += ["gas_mass_balance_residual_mearth", "gas_mass_balance_residual_fraction"]
    base += ["dt_min_yr", "dt_max_yr", "dt_mean_yr", "n_steps", "n_advective_limited", "n_diffusive_limited", "n_max_timestep_limited", "n_output_limited", "n_gas_flux_limited"]
    return base


def initialize_diagnostics(path: Path) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=diagnostics_fieldnames()).writeheader()

def append_diagnostics(
    path: Path, snap_idx: int, step: int, t: float, dt: float,
    state: Dict[str, np.ndarray], grid: Dict[str, np.ndarray], disk: Dict[str, np.ndarray],
    br_diag: Dict[str, np.ndarray], carrier_coeff: Dict[str, np.ndarray],
    runtime: Dict[str, Any], release: Dict[str, np.ndarray],
) -> None:
    area = grid["area"]
    mass_field = lambda name: annulus_integral_sigma(area, state[name])
    mass_array = lambda arr: annulus_integral_sigma(area, arr)
    CO_peb = state[co_pure_field("pebble")] + state[co_at_co2_field("pebble")] + state[co_at_h2o_field("pebble")]
    CO_small = state[co_pure_field("small")] + state[co_at_co2_field("small")] + state[co_at_h2o_field("small")]
    CO_hidden = state[co_at_co2_field("pebble")] + state[co_at_h2o_field("pebble")] + state[co_at_co2_field("small")] + state[co_at_h2o_field("small")]
    CO2_peb = state[co2_pure_field("pebble")] + state[co2_at_h2o_field("pebble")]
    CO2_small = state[co2_pure_field("small")] + state[co2_at_h2o_field("small")]
    H2O_peb, H2O_small = state[h2o_field("pebble")], state[h2o_field("small")]
    current = {"CO": mass_array(total_CO(state)), "CO2": mass_array(total_CO2(state)), "H2O": mass_array(total_H2O(state))}
    initial = runtime["initial_mass_g"]
    scalar = runtime["scalar"]
    cap_init, cap_cum = runtime["capacity_initial"], runtime["capacity_cumulative"]
    cap_current = runtime.get("capacity_current", {})

    def radial_mass(values: Dict[str, np.ndarray], key: str) -> float:
        return float(np.sum(area * values.get(key, np.zeros_like(area))))
    def release_mass(key: str) -> float:
        arr = release.get(key)
        return float(np.sum(arr)) if arr is not None else 0.0
    def weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
        w = np.maximum(np.asarray(weights, dtype=float), 0.0)
        return float(np.sum(w * values) / max(float(np.sum(w)), 1.0e-300))
    def rel_velocity(v: np.ndarray, v0: np.ndarray) -> np.ndarray:
        floor = max(1.0e-30, 1.0e-12 * float(np.nanmax(np.abs(v0))))
        return np.abs(v - v0) / np.maximum(np.abs(v0), floor)

    eps_peb = np.asarray(br_diag.get("epsilon_pebble", np.zeros_like(area)), dtype=float)
    peb_weights = area * carrier_total_solid_sigma(state, "pebble")
    gas_rel = rel_velocity(disk["v_g"], disk.get("v_g_no_backreaction", disk["v_g"]))
    peb_rel = rel_velocity(carrier_coeff["pebble"]["v"], carrier_coeff["pebble"].get("v_no_backreaction", carrier_coeff["pebble"]["v"]))

    row = {key: 0.0 for key in diagnostics_fieldnames()}
    row.update({
        "snapshot": snap_idx, "step": step, "time_yr": t / YR, "dt_yr": dt / YR,
        "M_gas_msun": annulus_integral_sigma(area, disk["Sigma_g"]) / MSUN,
        "M_gas_fixed_msun": annulus_integral_sigma(area, disk["Sigma_g"]) / MSUN,
        "M_ref_pebble_mearth": mass_field(ref_field("pebble")) / MEARTH,
        "M_ref_small_mearth": mass_field(ref_field("small")) / MEARTH,
        "M_CO_total_mearth": current["CO"] / MEARTH, "M_CO_gas_mearth": mass_field("CO_gas") / MEARTH,
        "M_CO_solid_mearth": mass_array(CO_peb + CO_small) / MEARTH, "M_CO_hidden_mearth": mass_array(CO_hidden) / MEARTH,
        "M_CO_matrix_mearth": mass_array(CO_hidden) / MEARTH, "M_CO_pebble_mearth": mass_array(CO_peb) / MEARTH, "M_CO_small_mearth": mass_array(CO_small) / MEARTH,
        "M_CO2_total_mearth": current["CO2"] / MEARTH, "M_CO2_gas_mearth": mass_field("CO2_gas") / MEARTH,
        "M_CO2_solid_mearth": mass_array(CO2_peb + CO2_small) / MEARTH, "M_CO2_pebble_mearth": mass_array(CO2_peb) / MEARTH, "M_CO2_small_mearth": mass_array(CO2_small) / MEARTH,
        "M_H2O_total_mearth": current["H2O"] / MEARTH, "M_H2O_gas_mearth": mass_field("H2O_gas") / MEARTH,
        "M_H2O_solid_mearth": mass_array(H2O_peb + H2O_small) / MEARTH, "M_H2O_pebble_mearth": mass_array(H2O_peb) / MEARTH, "M_H2O_small_mearth": mass_array(H2O_small) / MEARTH,
        "retained_CO_fraction": current["CO"] / max(initial["CO"], 1.0e-300), "retained_CO2_fraction": current["CO2"] / max(initial["CO2"], 1.0e-300), "retained_H2O_fraction": current["H2O"] / max(initial["H2O"], 1.0e-300),
        "min_sigma": min(float(np.min(state[k])) for k in ALL_FIELDS), "max_sigma": max(float(np.max(state[k])) for k in ALL_FIELDS),
        "max_epsilon_pebble": float(np.nanmax(eps_peb)), "median_epsilon_pebble": float(np.nanmedian(eps_peb)), "mass_weighted_epsilon_pebble": weighted_mean(eps_peb, peb_weights),
        "max_backreaction_X": float(np.nanmax(disk.get("backreaction_X", [0]))), "median_backreaction_X": float(np.nanmedian(disk.get("backreaction_X", [0]))),
        "max_backreaction_Y": float(np.nanmax(disk.get("backreaction_Y", [0]))), "median_backreaction_Y": float(np.nanmedian(disk.get("backreaction_Y", [0]))),
        "max_backreaction_A": float(np.nanmax(disk.get("backreaction_A", [0]))), "median_backreaction_A": float(np.nanmedian(disk.get("backreaction_A", [0]))),
        "max_backreaction_B": float(np.nanmax(disk.get("backreaction_B", [0]))), "median_backreaction_B": float(np.nanmedian(disk.get("backreaction_B", [0]))),
        "min_vr_g": float(np.nanmin(disk["v_g"])), "max_vr_g": float(np.nanmax(disk["v_g"])), "min_pebble_v": float(np.nanmin(carrier_coeff["pebble"]["v"])), "max_pebble_v": float(np.nanmax(carrier_coeff["pebble"]["v"])),
        "max_rel_delta_v_gas_backreaction": float(np.nanmax(gas_rel)), "mass_weighted_rel_delta_v_gas_backreaction": weighted_mean(gas_rel, area * disk["Sigma_g"]),
        "max_rel_delta_v_pebble_backreaction": float(np.nanmax(peb_rel)), "mass_weighted_rel_delta_v_pebble_backreaction": weighted_mean(peb_rel, peb_weights),
        "dt_min_yr": (runtime["dt_min_s"] / YR if np.isfinite(runtime["dt_min_s"]) else 0.0), "dt_max_yr": runtime["dt_max_s"] / YR,
        "dt_mean_yr": (runtime["dt_sum_s"] / max(runtime["n_steps"], 1)) / YR, "n_steps": runtime["n_steps"],
        "n_advective_limited": runtime["limiter_counts"].get("advective", 0), "n_diffusive_limited": runtime["limiter_counts"].get("diffusive", 0),
        "n_max_timestep_limited": runtime["limiter_counts"].get("max_timestep", 0), "n_output_limited": runtime["limiter_counts"].get("output", 0),
        "n_gas_flux_limited": runtime["limiter_counts"].get("gas_flux", 0),
    })

    for target in ("CO", "CO2", "H2O", "pebble_solids", "small_solids", "gas_disk"):
        row[f"cum_boundary_inner_{target}_mearth"] = scalar.get(f"boundary_inner_{target}_g", 0.0) / MEARTH
        row[f"cum_boundary_outer_{target}_mearth"] = scalar.get(f"boundary_outer_{target}_g", 0.0) / MEARTH
    for species in ("CO", "CO2", "H2O", "refractory", "gas_disk"):
        row[f"cum_clipped_added_{species}_mearth"] = scalar.get(f"clipped_added_{species}_g", 0.0) / MEARTH
        row[f"max_negative_{species}"] = scalar.get(f"max_negative_{species}", 0.0)
        row[f"clip_events_{species}"] = scalar.get(f"clip_events_{species}", 0.0)

    for species in ("CO", "CO2", "H2O"):
        gain = release_mass(f"cum_dM_{species}_gas_gain") / MEARTH
        loss = release_mass(f"cum_dM_{species}_gas_loss") / MEARTH
        net = release_mass(f"cum_dM_{species}_gas") / MEARTH
        row[f"cum_phase_gas_gain_{species}_mearth"] = gain
        row[f"cum_phase_gas_loss_{species}_mearth"] = loss
        row[f"cum_phase_gas_net_{species}_mearth"] = net
        row[f"phase_cycling_factor_{species}"] = (gain + loss) / max(abs(net), 1.0e-30)
        row[f"phase_max_abs_error_{species}"] = runtime["phase_errors"].get(f"phase_max_abs_error_{species}", 0.0)
        row[f"phase_max_rel_error_{species}"] = runtime["phase_errors"].get(f"phase_max_rel_error_{species}", 0.0)
        boundary = scalar.get(f"boundary_inner_{species}_g", 0.0) + scalar.get(f"boundary_outer_{species}_g", 0.0)
        clipped = scalar.get(f"clipped_added_{species}_g", 0.0)
        residual = current[species] + boundary - initial[species] - clipped
        row[f"mass_balance_residual_{species}_mearth"] = residual / MEARTH
        row[f"mass_balance_residual_fraction_{species}"] = residual / max(initial[species], 1.0e-300)

    # Independent bulk-gas mass ledger (outflow and floor additions).
    initial_gas = runtime["initial_gas_mass_g"]
    current_gas = annulus_integral_sigma(area, disk["Sigma_g"])
    gas_lost = (scalar.get("boundary_inner_gas_disk_g", 0.0)
                + scalar.get("boundary_outer_gas_disk_g", 0.0))
    gas_floor_added = scalar.get("clipped_added_gas_disk_g", 0.0)
    gas_residual = current_gas + gas_lost - initial_gas - gas_floor_added
    row["gas_mass_balance_residual_mearth"] = gas_residual / MEARTH
    row["gas_mass_balance_residual_fraction"] = gas_residual / max(initial_gas, 1.0e-300)

    for channel in ("CO_at_CO2", "CO_at_H2O", "CO2_at_H2O"):
        key = f"dSigma_capacity_excess_{channel}"
        mi, mcur, mc = radial_mass(cap_init, key), radial_mass(cap_current, key), radial_mass(cap_cum, key)
        row[f"initial_capacity_excess_{channel}_mearth"] = mi / MEARTH
        row[f"current_capacity_excess_{channel}_mearth"] = mcur / MEARTH
        arr_cur = np.asarray(cap_current.get(key, np.zeros_like(area)), dtype=float)
        active_threshold = max(1.0e-300, 1.0e-12 * float(np.nanmax(arr_cur)) if np.any(arr_cur > 0.0) else 1.0e-300)
        row[f"current_capacity_active_cell_fraction_{channel}"] = float(np.mean(arr_cur > active_threshold))
        row[f"cum_capacity_excess_{channel}_mearth"] = mc / MEARTH
        row[f"total_capacity_excess_{channel}_mearth"] = (mi + mc) / MEARTH
    for species in ("CO", "CO2"):
        key = f"dSigma_capacity_to_gas_{species}"
        mi, mcur, mc = radial_mass(cap_init, key), radial_mass(cap_current, key), radial_mass(cap_cum, key)
        row[f"initial_capacity_to_gas_{species}_mearth"] = mi / MEARTH
        row[f"current_capacity_to_gas_{species}_mearth"] = mcur / MEARTH
        row[f"cum_capacity_to_gas_{species}_mearth"] = mc / MEARTH
        row[f"total_capacity_to_gas_{species}_mearth"] = (mi + mc) / MEARTH

    with open(path, "a", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=diagnostics_fieldnames()).writerow(row)

# -----------------------------
# Main run
# -----------------------------
def write_outputs(
    output_dir: Path, snap_idx: int, step: int, t: float, dt: float,
    state: Dict[str, np.ndarray], grid: Dict[str, np.ndarray], disk: Dict[str, np.ndarray],
    carrier_coeff: Dict[str, Dict[str, np.ndarray]], vertical: Dict[str, Any],
    params: Dict[str, Any], diag_path: Path, release: Dict[str, np.ndarray],
    br_diag: Dict[str, np.ndarray], runtime: Dict[str, Any],
) -> None:
    capacity_current: Dict[str, np.ndarray] = {}
    build_phase_targets(
        state, vertical, params, capacity_diagnostics=capacity_current
    )
    runtime["capacity_current"] = capacity_current
    output_fields = dict(release)
    output_fields.update(capacity_output_fields(runtime["capacity_initial"], runtime["capacity_cumulative"], grid, current=capacity_current))
    if bool(params["output"]["save_1d_csv"]):
        write_snapshot_1d(output_dir, snap_idx, t, state, grid, disk, carrier_coeff, vertical, output_fields, params=params)
    append_diagnostics(diag_path, snap_idx, step, t, dt, state, grid, disk, br_diag, carrier_coeff, runtime, output_fields)
    if bool(params["output"]["save_2d_npz"]):
        every = int(params["output"]["save_2d_every_n_snapshots"])
        if every <= 1 or snap_idx % every == 0:
            write_snapshot_2d(output_dir, snap_idx, t, state, grid, disk, carrier_coeff, vertical)

def run(params: Dict[str, Any]) -> None:
    output_dir = Path(params["simulation"]["output_dir"]).expanduser().resolve()
    ensure_output_dir(output_dir, overwrite=bool(params["simulation"]["overwrite"]), save_2d=bool(params["output"]["save_2d_npz"]))
    write_resolved_params(params, output_dir)
    grid = make_grid(params)
    disk = build_fixed_disk(grid, params)
    carrier_coeff = build_carrier_coefficients(grid, disk, params)
    vertical = make_vertical_structure(grid, disk, carrier_coeff, params)
    capacity_initial: Dict[str, np.ndarray] = {}
    state = initialize_state(grid, disk, vertical, params, capacity_diagnostics=capacity_initial)

    t_end = float(params["simulation"]["t_end_yr"]) * YR
    save_interval = (float(params["simulation"]["save_interval_fraction"]) * t_end if params["simulation"]["save_interval_yr"] is None else float(params["simulation"]["save_interval_yr"]) * YR)
    save_interval = max(save_interval, 1.0e-30)
    diag_path = output_dir / "diagnostics.csv"
    initialize_diagnostics(diag_path)
    update_gas = bool(params["gas"].get("update_gas", False))
    backreaction = get_backreaction_enabled(params)
    gas_mode = params["gas"].get("evolution_mode", "viscous_diffusion")
    full_gas = update_gas and gas_mode == "viscous_diffusion"

    area = grid["area"]
    runtime: Dict[str, Any] = {
        "initial_mass_g": {"CO": annulus_integral_sigma(area, total_CO(state)), "CO2": annulus_integral_sigma(area, total_CO2(state)), "H2O": annulus_integral_sigma(area, total_H2O(state))},
        "initial_gas_mass_g": annulus_integral_sigma(area, disk["Sigma_g"]),
        "scalar": {}, "capacity_initial": capacity_initial, "capacity_cumulative": {}, "phase_errors": {},
        "dt_min_s": np.inf, "dt_max_s": 0.0, "dt_sum_s": 0.0, "n_steps": 0,
        "limiter_counts": {"advective": 0, "diffusive": 0, "max_timestep": 0, "output": 0, "gas_flux": 0},
    }
    t = 0.0; step = 0; snap_idx = 0; next_save = 0.0; last_dt = 0.0; release_interval_dt = 0.0
    release_since_output = zero_release_integral(grid)
    release_cumulative = zero_release_integral(grid)

    # The initial output and first CFL decision must describe exactly the same
    # physical state, including the initial backreaction velocities.
    disk = refresh_disk_after_sigma_update(grid, disk, params)
    carrier_coeff = build_carrier_coefficients(grid, disk, params)
    br_diag: Dict[str, np.ndarray] = {}
    if backreaction:
        disk, carrier_coeff, br_diag = apply_dust_backreaction_to_velocities(state, disk, carrier_coeff, params)
    vertical = make_vertical_structure(grid, disk, carrier_coeff, params)
    initial_release = release_output_fields(release_since_output, release_cumulative, release_interval_dt, grid)
    write_outputs(output_dir, snap_idx, step, t, last_dt, state, grid, disk, carrier_coeff, vertical, params, diag_path, initial_release, br_diag, runtime)
    snap_idx += 1; next_save += save_interval

    progress_every = int(params["simulation"]["progress_every"])
    mode_desc = (gas_mode if update_gas else "fixed_surface_density")
    print(f"Writing outputs to: {output_dir}")
    print(f"t_end = {t_end / YR:.6g} yr, save_interval = {save_interval / YR:.6g} yr")
    print("1+1D model: radial transport with vertically averaged snow-surface phase terms; " + f"{mode_desc} H2/He gas disk.")
    if params["gas"].get("gap", {}).get("enabled", False):
        gp = params["gas"]["gap"]
        eta = disk["eta"]
        # eta changes sign from positive to negative at a pressure minimum;
        # negative to positive at the outer-edge pressure maximum.
        pressure_maxima = np.where((eta[:-1] < 0.0) & (eta[1:] >= 0.0))[0]
        center = float(gp["center_au"])
        if pressure_maxima.size:
            idx = pressure_maxima[np.argmin(np.abs(grid["r_au"][pressure_maxima] - center))]
            trap = 0.5 * (grid["r_au"][idx] + grid["r_au"][idx+1])
            print(f"Prescribed Gaussian viscosity gap: center={center:g} au, sigma={disk['gap_width_au']:.3g} au, peak factor={float(gp['viscosity_factor']):g}; initial pressure maximum ~{trap:.3g} au")
        else:
            print(f"WARNING: Gaussian viscosity gap at {center:g} au has NO initial pressure maximum; inspect eta(r) and gap strength/width.", file=sys.stderr)
    pcfg = params["dust"]["phase_partition"]
    print("Phase model: condensation_weighting=" + str(pcfg["condensation_weighting"])
          + ", capture_mode=" + str(pcfg.get("condensation_capture_mode", "max_survival"))
          + ", preserve_matrix_history=" + str(pcfg.get("preserve_matrix_history", True))
          + ", prelimit_capacity=" + str(pcfg.get("prelimit_condensation_by_capacity", True))
          + ", ice_loss_mode=" + str(pcfg.get("ice_loss_mode", "mixing_limited")))

    while t < t_end * (1.0 - 1.0e-14):
        # All gas/dust velocities now describe state(t) *before* selecting dt.
        # Do not recalculate velocities between CFL and transport. Both gas and
        # transported solids/vapors advance using the same old-state dynamics.
        raw_dt, dt_info = compute_timestep(grid, disk, carrier_coeff, params, return_diagnostics=True)
        dt = min(raw_dt, t_end - t, next_save - t if next_save > t else raw_dt)
        limiter = dt_info["limiter"]
        if dt < raw_dt * (1.0 - 1.0e-12):
            limiter = "output"
        if dt <= 0.0:
            dt = min(raw_dt, t_end - t)

        trial_gas_sigma = None
        gas_flux = None
        gas_diag = {}
        if full_gas:
            # The viscous part is solved implicitly. The feedback-pressure
            # term and vapor advection are explicit; bound the *actual* gas
            # mass flux so no tracer donor cell loses more than its inventory.
            # Retrying shrinks dt BEFORE any state or disk field is changed.
            for attempt in range(24):
                try:
                    trial_gas_sigma, gas_flux, gas_diag = propose_gas_viscous_step(
                        disk, dt, grid, params
                    )
                    dt_flux = gas_flux_cfl_time(
                        disk["Sigma_g"], gas_flux, grid,
                        float(params["numerics"]["cfl_advective"])
                    )
                    if dt <= dt_flux * (1.0 + 1.0e-12):
                        break
                    dt = min(dt * 0.5, dt_flux * 0.9)
                except FloatingPointError:
                    dt *= 0.5
                limiter = "gas_flux"
                if not np.isfinite(dt) or dt <= 0.0:
                    raise RuntimeError("Gas-flux timestep fell to zero")
            else:
                raise RuntimeError("Could not find stable gas-flux timestep after 24 attempts")

        # Transport all species using precisely the velocity and Sigma_g that
        # set dt. In full viscous mode vapors use the gas mass flux from the
        # conservative gas solve, retaining a uniform vapor/gas mixing ratio.
        state, transport_diag = transport_step(
            state, dt, grid, disk, carrier_coeff, params, return_diagnostics=True,
            gas_flux_edges=gas_flux,
        )
        accumulate_scalar_diagnostics(runtime["scalar"], transport_diag)

        if update_gas:
            if full_gas:
                disk["Sigma_g"] = trial_gas_sigma
            else:
                disk, gas_diag = update_gas_surface_density_prescribed(
                    disk, dt, grid, params, return_diagnostics=True
                )
            accumulate_scalar_diagnostics(runtime["scalar"], gas_diag)

        # Reconstruct the vertical chemistry using gas at t+dt, then exchange
        # phases. Backreaction is refreshed afterward to include new solids.
        disk = refresh_disk_after_sigma_update(grid, disk, params)
        carrier_coeff = build_carrier_coefficients(grid, disk, params)
        if backreaction:
            disk, carrier_coeff, br_diag = apply_dust_backreaction_to_velocities(state, disk, carrier_coeff, params)
        vertical = make_vertical_structure(grid, disk, carrier_coeff, params)

        before_phase = {key: value.copy() for key, value in state.items()}
        state, capacity_step = phase_relaxation_step(state, dt, vertical, params, return_diagnostics=True)
        phase_errors = check_phase_conservation(before_phase, state, params)
        for key, value in phase_errors.items():
            runtime["phase_errors"][key] = max(runtime["phase_errors"].get(key, 0.0), float(value))
        accumulate_array_diagnostics(runtime["capacity_cumulative"], capacity_step)

        step_release = compute_release(before_phase, state, dt, grid)
        accumulate_release(release_since_output, step_release)
        accumulate_release(release_cumulative, step_release)
        release_interval_dt += dt
        t += dt; step += 1; last_dt = dt

        # Phase exchange changes dust mass and hence backreaction. Refresh
        # velocity diagnostics before the next CFL estimate and output.
        if backreaction:
            disk = refresh_disk_after_sigma_update(grid, disk, params)
            carrier_coeff = build_carrier_coefficients(grid, disk, params)
            disk, carrier_coeff, br_diag = apply_dust_backreaction_to_velocities(state, disk, carrier_coeff, params)

        runtime["dt_min_s"] = min(runtime["dt_min_s"], dt)
        runtime["dt_max_s"] = max(runtime["dt_max_s"], dt)
        runtime["dt_sum_s"] += dt
        runtime["n_steps"] += 1
        runtime["limiter_counts"][limiter] = runtime["limiter_counts"].get(limiter, 0) + 1

        if t >= next_save - 1.0e-9 * save_interval or t >= t_end * (1.0 - 1.0e-14):
            fields = release_output_fields(release_since_output, release_cumulative, release_interval_dt, grid)
            write_outputs(output_dir, snap_idx, step, t, last_dt, state, grid, disk, carrier_coeff, vertical, params, diag_path, fields, br_diag, runtime)
            snap_idx += 1; next_save += save_interval
            release_since_output = zero_release_integral(grid); release_interval_dt = 0.0

        if progress_every > 0 and step % progress_every == 0:
            print(f"step={step:8d} t={t / YR:12.5e} yr dt={last_dt / YR:10.3e} yr snap={snap_idx - 1}", flush=True)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="1+1D mixed-ice volatile transport with full viscous or prescribed gas evolution."
    )
    parser.add_argument("params", type=str, help="Path to YAML parameter file.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    params = load_params(args.params)
    run(params)


if __name__ == "__main__":
    main()
