"""
analyze_run.py

Visual analysis for outputs from mixed_ice_transport_1p1d.py.

Usage:
    python analyze_run.py mixed_ice_1p1d_outputs_example
    python analyze_run.py mixed_ice_1p1d_outputs_example --snap latest
    python analyze_run.py mixed_ice_1p1d_outputs_example --snap 10 --skip-2d

Outputs are written by default to:
    <output_dir>/analysis_plots/

This script reads:
    diagnostics.csv
    snapshots/snapshot_*.csv
    snapshots_2d/snapshot2d_*.npz  [optional]

It makes:
    diagnostics_masses.png
    diagnostics_fractions.png
    final_volatile_profiles.png
    final_carrier_profiles.png
    final_c_o_profiles.png
    final_snow_surfaces.png
    time_radius_*.png
    selected_2d_*.png
    summary_metrics.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


from colorspacious import cspace_converter
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib import rc as mplrc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm, SymLogNorm


G = 6.67430e-8
KB = 1.380649e-16
M_H = 1.6735575e-24
AU = 1.495978707e13
YR = 365.25 * 24.0 * 3600.0
MSUN = 1.98847e33
MEARTH = 5.9722e27
PI = np.pi

def hex_to_rgb(value):
    """Convert hex to RGB values in the range [0, 1]."""
    value = value.lstrip('#')
    lv = len(value)
    return tuple(int(value[i:i + lv // 3], 16) / 255. for i in range(0, lv, lv // 3))

def create_perceptually_uniform_cmap(start_color: list, end_color: list, N: int = 256, return_color: bool = False):

    if type(start_color) is str:
        start_color = hex_to_rgb(start_color) if "#" in start_color else mcolors.to_rgb(start_color)
    if type(end_color) is str:
        end_color = hex_to_rgb(end_color) if "#" in end_color else mcolors.to_rgb(end_color)
    # Convert the start and end colors from RGB to LAB color space
    converter = cspace_converter("sRGB1", "CAM02-UCS")
    start_color_lab = converter(start_color)
    end_color_lab = converter(end_color)
    
    # Create a linear interpolation of colors in LAB color space
    lab_colors = np.linspace(start_color_lab, end_color_lab, N)
    
    # Convert the interpolated colors back to RGB
    converter = cspace_converter("CAM02-UCS", "sRGB1")
    rgb_colors = converter(lab_colors)
    
    # Ensure all RGB values are within the valid range [0, 1]
    rgb_colors = np.clip(rgb_colors, 0, 1)
    if return_color:
        return rgb_colors
    
    # Create and return the colormap
    return LinearSegmentedColormap.from_list("custom_colormap", rgb_colors)

def create_diverging_cmap(start_color: list, middle_color: list, end_color: list, N: int = 256):

    first_map = create_perceptually_uniform_cmap(start_color, middle_color, N=N, return_color=True)
    second_map = create_perceptually_uniform_cmap(middle_color, end_color, N=N, return_color=True)

    # combine them and build a new colormap
    combined = np.vstack((first_map, second_map))

    return LinearSegmentedColormap.from_list("diverging_colormap", combined)
    


def get_color_list(color_list: list, i: int):

    if i > len(color_list) - 1:
        other_colors = mcolors.CSS4_COLORS
        rand_color_index = np.random.randint(0, high=len(other_colors))
        this_color = list(mcolors.CSS4_COLORS.values())[rand_color_index]
        color_list.append(this_color)

    return color_list

c0 = [0.52156863, 0.58823529, 0.84313725]
c1 = [0.96078431, 0.93333333, 0.37254902]
c2 = [0.75686275, 0.21176471, 0.11372549]
N_samples = 256
temp_cmap = create_diverging_cmap(c0, c1, c2, N=N_samples)

color_list = [[0.75686275, 0.21176471, 0.11372549],
              [0.52156863, 0.58823529, 0.84313725],
              [0.41568627, 0.24313725, 0.42352941],
              [0.85882353, 0.58039216, 0.05098039],
              [0.15294118, 0.19607843, 0.23529412],
              [0.50196078, 0.52941176, 0.50196078],
              [0.74509804, 0.52156863, 0.56862745],
             ]

# -------------------------
# Functions
# -------------------------

EPS = 1.0e-300

MW = {
    "CO": 28.0101,
    "CO2": 44.0095,
    "H2O": 18.01528,
}


@dataclass
class SnapshotInfo:
    index: int
    path: Path


def parse_snapshot_index(path: Path) -> int:
    m = re.search(r"(_\d+)", path.stem)
    if m is None:
        raise ValueError(f"Could not parse snapshot index from {path}")
    return int(m.group(1)[1:])


def list_snapshots(output_dir: Path) -> List[SnapshotInfo]:
    paths = sorted(Path(f"{output_dir}/snapshots").glob("snapshot_*.csv"))
    return [SnapshotInfo(parse_snapshot_index(p), p) for p in paths]


def list_snapshots_2d(output_dir: Path) -> List[SnapshotInfo]:
    snap_dir = Path(f"{output_dir}/snapshots_2d")
    if not snap_dir.exists():
        return []
    paths = sorted(snap_dir.glob("snapshot2d_*.npz"))
    return [SnapshotInfo(parse_snapshot_index(p), p) for p in paths]


def select_snapshot(snapshots: Sequence[SnapshotInfo], selector: str) -> SnapshotInfo:
    if not snapshots:
        raise FileNotFoundError("No snapshots found.")
    if selector == "latest":
        return snapshots[-1]
    if selector == "first":
        return snapshots[0]
    if selector == "middle":
        return snapshots[len(snapshots) // 2]
    idx = int(selector)
    for s in snapshots:
        if s.index == idx:
            return s
    raise ValueError(f"Snapshot index {idx} not found.")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


def has_columns(df: pd.DataFrame, cols: Iterable[str]) -> bool:
    return all(c in df.columns for c in cols)


def safe_log10(x: np.ndarray, floor: Optional[float] = None) -> np.ndarray:
    arr = np.asarray(x, dtype=float)
    if floor is None:
        positive = arr[np.isfinite(arr) & (arr > 0)]
        floor = 1.0e-300 if positive.size == 0 else max(np.nanmin(positive) * 1.0e-3, 1.0e-300)
    return np.log10(np.maximum(arr, floor))


def read_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def read_diagnostics(output_dir: Path) -> Optional[pd.DataFrame]:
    path = Path(f"{output_dir}/diagnostics.csv")
    if not path.exists():
        return None
    return pd.read_csv(path)


def read_2d_snapshot(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# -----------------------------
# Diagnostics plots
# -----------------------------
def make_diagnostics_plots(diag: Optional[pd.DataFrame], analysis_dir: Path) -> None:
    if diag is None or diag.empty or "time_yr" not in diag.columns:
        return

    mass_cols = [
        "M_CO_total_mearth", "M_CO_gas_mearth", "M_CO_solid_mearth", "M_CO_hidden_mearth",
        "M_CO2_total_mearth", "M_CO2_gas_mearth", "M_CO2_solid_mearth",
        "M_H2O_total_mearth", "M_H2O_gas_mearth", "M_H2O_solid_mearth",
    ]
    mass_labels = [
        "CO total", "CO gas", "CO solid", "CO hidden",
        "CO2 total", "CO2 gas", "CO2 solid",
        "H2O total", "H2O gas", "H2O solid",
    ]

    plt.figure(figsize=(9, 5))
    this_color_list = color_list[:]
    for i, (col, label) in enumerate(zip(mass_cols, mass_labels)):
        if col in enumerate(diag.columns):
            this_color_list = get_color_list(this_color_list, i)
            plt.plot(diag["time_yr"], np.maximum(diag[col], EPS), label=label, color=this_color_list[i])
    plt.yscale("log")
    plt.xlabel("Time [yr]")
    plt.ylabel("Mass [Earth masses]")
    plt.title("Global volatile inventories")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/diagnostics_masses.png")

    if has_columns(diag, ["M_CO_total_mearth", "M_CO_gas_mearth", "M_CO_hidden_mearth"]):
        total = np.maximum(diag["M_CO_total_mearth"].to_numpy(), EPS)
        plt.figure(figsize=(8, 5))
        plt.plot(diag["time_yr"], diag["M_CO_gas_mearth"] / total, label="CO gas / total CO", color=color_list[0])
        plt.plot(diag["time_yr"], diag["M_CO_hidden_mearth"] / total, label="hidden CO / total CO", color=color_list[1])
        if "M_CO_solid_mearth" in diag.columns:
            plt.plot(diag["time_yr"], diag["M_CO_solid_mearth"] / total, label="solid CO / total CO", color=color_list[2])
        plt.xlabel("Time [yr]")
        plt.ylabel("Fraction")
        plt.ylim(-0.02, 1.02)
        plt.title("Global CO partitioning")
        plt.legend()
        plt.grid(True, alpha=0.3)
        savefig(f"{analysis_dir}/diagnostics_fractions.png")


def _zero_like_any(data):
    """Return a zero array with the same shape as the first ndarray in data."""
    for v in data.values():
        if isinstance(v, np.ndarray):
            return np.zeros_like(v, dtype=float)
    raise ValueError("No ndarray found in data.")


def _get(data, key, missing="zero"):
    """Fetch a field from a snapshot dictionary."""
    if key in data:
        return np.asarray(data[key], dtype=float)

    if missing == "zero":
        return _zero_like_any(data)

    raise KeyError(f"Missing required field: {key}")


def _sum_fields(data, keys, missing="zero"):
    out = _zero_like_any(data)
    for key in keys:
        out = out + _get(data, key, missing=missing)
    return out

#### Volatile budgets #####

def compute_volatile_budgets(data, collapse_vertical=False):
    """
    Compute volatile budgets from a 1+1D mixed-ice snapshot dictionary.

    Parameters
    ----------
    data : dict
        Usually loaded from snapshot2d_XXXXXX.npz, e.g.
            with np.load(path) as f:
                data = {k: f[k] for k in f.files}

        Can also work with a dictionary of 1D arrays if the keys are adapted.

    collapse_vertical : bool
        If True and arrays are 2D, sum over vertical bins to return radial
        surface-density profiles. If False, return arrays in their native shape.

    Returns
    -------
    out : dict
        Dictionary containing CO, CO2, H2O budgets, carrier volatile budgets,
        carrier fractions, and molar C/O ratios.
    """

    # -------------------------
    # CO reservoirs
    # -------------------------
    co_pebble = _sum_fields(data, [
        "surfbin_CO_pure_ice_pebble",
        "surfbin_CO_at_CO2_ice_pebble",
        "surfbin_CO_at_H2O_ice_pebble",
    ])

    co_small = _sum_fields(data, [
        "surfbin_CO_pure_ice_small",
        "surfbin_CO_at_CO2_ice_small",
        "surfbin_CO_at_H2O_ice_small",
    ])

    co_gas = _get(data, "surfbin_CO_gas")

    co_hidden_pebble = _sum_fields(data, [
        "surfbin_CO_at_CO2_ice_pebble",
        "surfbin_CO_at_H2O_ice_pebble",
    ])

    co_hidden_small = _sum_fields(data, [
        "surfbin_CO_at_CO2_ice_small",
        "surfbin_CO_at_H2O_ice_small",
    ])

    # -------------------------
    # CO2 reservoirs
    # -------------------------
    co2_pebble = _sum_fields(data, [
        "surfbin_CO2_pure_ice_pebble",
        "surfbin_CO2_at_H2O_ice_pebble",
    ])

    co2_small = _sum_fields(data, [
        "surfbin_CO2_pure_ice_small",
        "surfbin_CO2_at_H2O_ice_small",
    ])

    co2_gas = _get(data, "surfbin_CO2_gas")

    # -------------------------
    # H2O reservoirs
    # -------------------------
    h2o_pebble = _get(data, "surfbin_H2O_ice_pebble")
    h2o_small = _get(data, "surfbin_H2O_ice_small")
    h2o_gas = _get(data, "surfbin_H2O_gas")

    # -------------------------
    # Optionally collapse z bins
    # -------------------------
    def collapse(x):
        if collapse_vertical and x.ndim == 2:
            return np.sum(x, axis=1)
        return x

    co_pebble = collapse(co_pebble)
    co_small = collapse(co_small)
    co_gas = collapse(co_gas)
    co_hidden_pebble = collapse(co_hidden_pebble)
    co_hidden_small = collapse(co_hidden_small)

    co2_pebble = collapse(co2_pebble)
    co2_small = collapse(co2_small)
    co2_gas = collapse(co2_gas)

    h2o_pebble = collapse(h2o_pebble)
    h2o_small = collapse(h2o_small)
    h2o_gas = collapse(h2o_gas)

    # -------------------------
    # Mass budgets
    # -------------------------
    co_total = co_gas + co_pebble + co_small
    co_hidden_total = co_hidden_pebble + co_hidden_small

    co2_total = co2_gas + co2_pebble + co2_small
    h2o_total = h2o_gas + h2o_pebble + h2o_small

    pebble_volatile_ice = co_pebble + co2_pebble + h2o_pebble
    small_volatile_ice = co_small + co2_small + h2o_small
    gas_volatile = co_gas + co2_gas + h2o_gas

    solid_volatile_ice = pebble_volatile_ice + small_volatile_ice
    total_volatile = solid_volatile_ice + gas_volatile

    # -------------------------
    # Molar carbon and oxygen budgets
    # -------------------------
    # These are proportional to mol cm^-2, or mol per vertical bin, depending
    # on the input units. Avogadro's number cancels in C/O ratios.
    n_co_pebble = co_pebble / MW["CO"]
    n_co_small = co_small / MW["CO"]
    n_co_gas = co_gas / MW["CO"]

    n_co2_pebble = co2_pebble / MW["CO2"]
    n_co2_small = co2_small / MW["CO2"]
    n_co2_gas = co2_gas / MW["CO2"]

    n_h2o_pebble = h2o_pebble / MW["H2O"]
    n_h2o_small = h2o_small / MW["H2O"]
    n_h2o_gas = h2o_gas / MW["H2O"]

    C_pebble = n_co_pebble + n_co2_pebble
    O_pebble = n_co_pebble + 2.0 * n_co2_pebble + n_h2o_pebble

    C_small = n_co_small + n_co2_small
    O_small = n_co_small + 2.0 * n_co2_small + n_h2o_small

    C_gas = n_co_gas + n_co2_gas
    O_gas = n_co_gas + 2.0 * n_co2_gas + n_h2o_gas

    C_solid = C_pebble + C_small
    O_solid = O_pebble + O_small

    C_total = C_solid + C_gas
    O_total = O_solid + O_gas

    out = {
        # CO budgets
        "co_pebble": co_pebble,
        "co_small": co_small,
        "co_gas": co_gas,
        "co_total": co_total,
        "co_hidden_pebble": co_hidden_pebble,
        "co_hidden_small": co_hidden_small,
        "co_hidden_total": co_hidden_total,

        # CO2 budgets
        "co2_pebble": co2_pebble,
        "co2_small": co2_small,
        "co2_gas": co2_gas,
        "co2_total": co2_total,

        # H2O budgets
        "h2o_pebble": h2o_pebble,
        "h2o_small": h2o_small,
        "h2o_gas": h2o_gas,
        "h2o_total": h2o_total,

        # Total volatile mass budgets
        "pebble_volatile_ice": pebble_volatile_ice,
        "small_volatile_ice": small_volatile_ice,
        "solid_volatile_ice": solid_volatile_ice,
        "gas_volatile": gas_volatile,
        "total_volatile": total_volatile,

        # Fractions of total CO
        "co_gas_fraction": co_gas / np.maximum(co_total, EPS),
        "co_pebble_fraction": co_pebble / np.maximum(co_total, EPS),
        "co_small_fraction": co_small / np.maximum(co_total, EPS),
        "co_hidden_fraction": co_hidden_total / np.maximum(co_total, EPS),

        # Fractions of solid volatile ice
        "pebble_fraction_of_solid_volatile_ice": (
            pebble_volatile_ice / np.maximum(solid_volatile_ice, EPS)
        ),
        "small_fraction_of_solid_volatile_ice": (
            small_volatile_ice / np.maximum(solid_volatile_ice, EPS)
        ),

        # Fractions of total volatile mass, including gas
        "pebble_fraction_of_total_volatile": (
            pebble_volatile_ice / np.maximum(total_volatile, EPS)
        ),
        "small_fraction_of_total_volatile": (
            small_volatile_ice / np.maximum(total_volatile, EPS)
        ),
        "gas_fraction_of_total_volatile": (
            gas_volatile / np.maximum(total_volatile, EPS)
        ),

        # Molar C/O ratios
        "C_over_O_pebble": C_pebble / np.maximum(O_pebble, EPS),
        "C_over_O_small": C_small / np.maximum(O_small, EPS),
        "C_over_O_solid": C_solid / np.maximum(O_solid, EPS),
        "C_over_O_gas": C_gas / np.maximum(O_gas, EPS),
        "C_over_O_total": C_total / np.maximum(O_total, EPS),

        # Raw molar C and O budgets, useful for debugging/masking
        "C_pebble": C_pebble,
        "O_pebble": O_pebble,
        "C_small": C_small,
        "O_small": O_small,
        "C_gas": C_gas,
        "O_gas": O_gas,
        "C_total": C_total,
        "O_total": O_total,
    }

    return out

def _col(df, name):
    """Return a DataFrame column or zeros if missing."""
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.zeros_like(df["r_au"].to_numpy(dtype=float))


def _dlnr_from_centers(r):
    """Approximate cell dlnr from cell-centered radii."""
    logr = np.log(np.asarray(r, dtype=float))
    log_edges = np.empty(len(logr) + 1)

    log_edges[1:-1] = 0.5 * (logr[:-1] + logr[1:])
    log_edges[0] = logr[0] - 0.5 * (logr[1] - logr[0])
    log_edges[-1] = logr[-1] + 0.5 * (logr[-1] - logr[-2])

    return np.diff(log_edges)


def add_volatile_budget_columns_1d(df):
    """
    Add carrier-resolved volatile budgets and C/O ratios to a 1D snapshot CSV.

    This expects columns from mixed_ice_transport_1p1d.py, e.g.
    CO_gas, CO_pure_ice_pebble, CO_at_CO2_ice_small, etc.
    """

    df = df.copy()

    # -------------------------
    # CO reservoirs
    # -------------------------
    co_pebble = (
        _col(df, "CO_pure_ice_pebble")
        + _col(df, "CO_at_CO2_ice_pebble")
        + _col(df, "CO_at_H2O_ice_pebble")
    )

    co_small = (
        _col(df, "CO_pure_ice_small")
        + _col(df, "CO_at_CO2_ice_small")
        + _col(df, "CO_at_H2O_ice_small")
    )

    co_gas = _col(df, "CO_gas")

    co_hidden_pebble = (
        _col(df, "CO_at_CO2_ice_pebble")
        + _col(df, "CO_at_H2O_ice_pebble")
    )

    co_hidden_small = (
        _col(df, "CO_at_CO2_ice_small")
        + _col(df, "CO_at_H2O_ice_small")
    )

    # -------------------------
    # CO2 reservoirs
    # -------------------------
    co2_pebble = (
        _col(df, "CO2_pure_ice_pebble")
        + _col(df, "CO2_at_H2O_ice_pebble")
    )

    co2_small = (
        _col(df, "CO2_pure_ice_small")
        + _col(df, "CO2_at_H2O_ice_small")
    )

    co2_gas = _col(df, "CO2_gas")

    # -------------------------
    # H2O reservoirs
    # -------------------------
    h2o_pebble = _col(df, "H2O_ice_pebble")
    h2o_small = _col(df, "H2O_ice_small")
    h2o_gas = _col(df, "H2O_gas")

    # -------------------------
    # Total budgets
    # -------------------------
    co_total = co_gas + co_pebble + co_small
    co_hidden_total = co_hidden_pebble + co_hidden_small

    co2_total = co2_gas + co2_pebble + co2_small
    h2o_total = h2o_gas + h2o_pebble + h2o_small

    pebble_volatile_ice = co_pebble + co2_pebble + h2o_pebble
    small_volatile_ice = co_small + co2_small + h2o_small

    solid_volatile_ice = pebble_volatile_ice + small_volatile_ice
    gas_volatile = co_gas + co2_gas + h2o_gas
    total_volatile = solid_volatile_ice + gas_volatile

    # -------------------------
    # Molar C/O budgets
    # -------------------------
    n_co_pebble = co_pebble / MW["CO"]
    n_co_small = co_small / MW["CO"]
    n_co_gas = co_gas / MW["CO"]

    n_co2_pebble = co2_pebble / MW["CO2"]
    n_co2_small = co2_small / MW["CO2"]
    n_co2_gas = co2_gas / MW["CO2"]

    n_h2o_pebble = h2o_pebble / MW["H2O"]
    n_h2o_small = h2o_small / MW["H2O"]
    n_h2o_gas = h2o_gas / MW["H2O"]

    C_pebble = n_co_pebble + n_co2_pebble
    O_pebble = n_co_pebble + 2.0 * n_co2_pebble + n_h2o_pebble

    C_small = n_co_small + n_co2_small
    O_small = n_co_small + 2.0 * n_co2_small + n_h2o_small

    C_gas = n_co_gas + n_co2_gas
    O_gas = n_co_gas + 2.0 * n_co2_gas + n_h2o_gas

    # -------------------------
    # Store columns
    # -------------------------
    df["co_pebble"] = co_pebble
    df["co_small"] = co_small
    df["co_gas_budget"] = co_gas
    df["co_total_budget"] = co_total
    df["co_hidden_total_budget"] = co_hidden_total

    df["co_gas_fraction_budget"] = co_gas / np.maximum(co_total, EPS)
    df["co_hidden_fraction_budget"] = co_hidden_total / np.maximum(co_total, EPS)
    df["co_pebble_fraction_budget"] = co_pebble / np.maximum(co_total, EPS)
    df["co_small_fraction_budget"] = co_small / np.maximum(co_total, EPS)

    df["co2_pebble"] = co2_pebble
    df["co2_small"] = co2_small
    df["co2_total_budget"] = co2_total

    df["h2o_pebble"] = h2o_pebble
    df["h2o_small"] = h2o_small
    df["h2o_total_budget"] = h2o_total

    df["pebble_volatile_ice_budget"] = pebble_volatile_ice
    df["small_volatile_ice_budget"] = small_volatile_ice
    df["solid_volatile_ice_budget"] = solid_volatile_ice
    df["gas_volatile_budget"] = gas_volatile
    df["total_volatile_budget"] = total_volatile

    df["pebble_fraction_of_solid_volatile_ice"] = (
        pebble_volatile_ice / np.maximum(solid_volatile_ice, EPS)
    )
    df["small_fraction_of_solid_volatile_ice"] = (
        small_volatile_ice / np.maximum(solid_volatile_ice, EPS)
    )

    df["pebble_fraction_of_total_volatile"] = (
        pebble_volatile_ice / np.maximum(total_volatile, EPS)
    )
    df["small_fraction_of_total_volatile"] = (
        small_volatile_ice / np.maximum(total_volatile, EPS)
    )
    df["gas_fraction_of_total_volatile"] = (
        gas_volatile / np.maximum(total_volatile, EPS)
    )

    df["C_over_O_pebble_budget"] = C_pebble / np.maximum(O_pebble, EPS)
    df["C_over_O_small_budget"] = C_small / np.maximum(O_small, EPS)
    df["C_over_O_gas_budget"] = C_gas / np.maximum(O_gas, EPS)

    # Mask low-mass C/O regions.
    small_thresh = 1e-8 * np.nanmax(small_volatile_ice)
    pebble_thresh = 1e-8 * np.nanmax(pebble_volatile_ice)

    df["C_over_O_small_masked"] = np.where(
        small_volatile_ice > small_thresh,
        df["C_over_O_small_budget"],
        np.nan,
    )

    df["C_over_O_pebble_masked"] = np.where(
        pebble_volatile_ice > pebble_thresh,
        df["C_over_O_pebble_budget"],
        np.nan,
    )

    return df


# -----------------------------
# 1D snapshot plots
# -----------------------------
def make_final_1d_plots(df: pd.DataFrame, analysis_dir: Path, snap_index: int) -> None:
    if "r_au" not in df.columns:
        return
    r = df["r_au"]

    plt.figure(figsize=(9, 5))
    i = 0
    this_color_list = color_list[:]
    for col, label in [
        ("CO_gas", "CO gas"), ("CO_solid_total", "CO solid"), ("CO_hidden_total", "hidden CO"),
        ("CO2_gas", "CO2 gas"), ("CO2_solid_total", "CO2 solid"),
        ("H2O_gas", "H2O gas"), ("H2O_solid_total", "H2O solid"),
    ]:
        if col in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.loglog(r, np.maximum(df[col], EPS), label=label, color=this_color_list[i])
            i += 1
    plt.xlabel("Radius [au]")
    plt.ylabel(r"Surface density [g cm$^{-2}$]")
    plt.title(f"Volatile reservoir profiles, snapshot {snap_index}")
    plt.ylim(bottom=1e-8)
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "final_volatile_profiles.png")

    plt.figure(figsize=(9, 5))
    i = 0
    this_color_list = color_list[:]
    for col, label in [
        ("CO_solid_pebble", "CO in pebbles"), ("CO_solid_small", "CO in small grains"),
        ("CO2_solid_pebble", "CO2 in pebbles"), ("CO2_solid_small", "CO2 in small grains"),
        ("H2O_solid_pebble", "H2O in pebbles"), ("H2O_solid_small", "H2O in small grains"),
    ]:
        if col in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.loglog(r, np.maximum(df[col], EPS), label=label, color=this_color_list[i])
            i += 1
    plt.xlabel("Radius [au]")
    plt.ylabel(r"Surface density [g cm$^{-2}$]")
    plt.ylim(bottom=1e-8)
    plt.title(f"Carrier-resolved solid volatiles, snapshot {snap_index}")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/final_carrier_profiles.png")

    plt.figure(figsize=(9, 5))
    i = 0
    this_color_list = color_list[:]
    for col, label in [
        ("C_over_O_gas", "gas C/O"),
        ("C_over_O_solid", "solid C/O"),
        ("C_over_O_pebble", "pebble C/O"),
        ("C_over_O_small", "small-grain C/O"),
        ("hidden_CO_fraction", "hidden CO fraction"),
        ("gas_CO_fraction", "gas CO fraction"),
    ]:
        if col in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.semilogx(r, df[col], label=label, color=this_color_list[i])
            i += 1
    plt.xlabel("Radius [au]")
    plt.ylabel("Ratio / fraction")
    plt.title(f"C/O and CO partitioning, snapshot {snap_index}")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/final_c_o_profiles.png")

    snow_cols = [c for c in df.columns if c.startswith("snow_surface_z_over_r_")]
    if snow_cols:
        i = 0
        this_color_list = color_list[:]
        plt.figure(figsize=(9, 5))
        for col in snow_cols:
            this_color_list = get_color_list(this_color_list, i)
            label = col.replace("snow_surface_z_over_r_", "")
            plt.semilogx(r, df[col], label=label, color=this_color_list[i])
            i += 1
        if "H_over_r" in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.semilogx(r, df["H_over_r"], label="H/r", color="gray", ls='--')
        plt.xlabel("Radius [au]")
        plt.ylabel("z/r")
        plt.title(f"Modeled snow surfaces, snapshot {snap_index}")
        plt.legend(ncols=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        savefig(f"{analysis_dir}/final_snow_surfaces.png")

    release_cols = [c for c in df.columns if c.startswith("dM")]
    if release_cols:
        i = 0
        this_color_list = color_list[:]
        plt.figure(figsize=(9, 5))
        for col in release_cols:
            label = col.replace("dM_", "")
            this_color_list = get_color_list(this_color_list, i)
            plt.semilogx(r, df[col]/MEARTH, label=label, color=this_color_list[i])
            i += 1
        plt.xlabel("Radius [au]")
        plt.ylabel(r"$dM_{\rm CO,release}$  [M$_{\oplus}$]")
        plt.title(f"CO mass release, snapshot {snap_index}")
        plt.legend(ncols=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        savefig(f"{analysis_dir}/co_release.png")


    co_pure = df["CO_pure_ice_pebble"] + df["CO_pure_ice_small"]
    co_at_h2o = df["CO_at_H2O_ice_small"] + df["CO_at_H2O_ice_pebble"]
    co_at_co2 = df["CO_at_CO2_ice_small"] + df["CO_at_CO2_ice_pebble"]
    co2_at_h2o = df["CO2_at_H2O_ice_small"] + df["CO2_at_H2O_ice_pebble"]
    plt.figure(figsize=(9, 5))
    plt.semilogx(r, co_pure, label="Pure CO Ice", color=color_list[0])
    plt.semilogx(r, co_at_co2, label="CO@CO2 Ice", color=color_list[1])
    plt.semilogx(r, co_at_h2o, label="CO@H2O Ice", color=color_list[2])
    plt.semilogx(r, co2_at_h2o, label="CO2@H2O Ice", color=color_list[3])
    plt.xlabel("Radius [au]")
    plt.ylabel(r"Surface density [g cm$^{-2}$]")
    plt.ylim(bottom=1e-8)
    plt.title(f"CO ices, snapshot {snap_index}")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/co_sigmas.png")

    co2_tot = df["CO2_pure_ice_small"] + df["CO2_pure_ice_pebble"] + df["CO2_at_H2O_ice_small"] + df["CO2_at_H2O_ice_pebble"]
    h2o_tot = df["H2O_ice_small"] + df["H2O_ice_pebble"]

    plt.semilogx(r, co_at_co2 / co2_tot, label="CO@CO2 Ice" , color=color_list[0])
    plt.semilogx(r, co_at_h2o / h2o_tot, label="CO@H2O Ice", color=color_list[1])
    plt.semilogx(r, co2_at_h2o / h2o_tot, label="CO2@H2O Ice", color=color_list[2])
    plt.xlabel("Radius [au]")
    plt.ylabel(r"Trapping capacity")
    plt.title(f"CO@X/X total, snapshot {snap_index}")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/co_trapping_capacity.png")


    floor = 1e-30

    co_pure    = df["CO_pure_ice_small"] + df["CO_pure_ice_pebble"]
    co_at_co2  = df["CO_at_CO2_ice_small"] + df["CO_at_CO2_ice_pebble"]
    co_at_h2o  = df["CO_at_H2O_ice_small"] + df["CO_at_H2O_ice_pebble"]
    co2_at_h2o = df["CO2_at_H2O_ice_small"] + df["CO2_at_H2O_ice_pebble"]
    
    co2_matrix_pure = df["CO2_pure_ice_small"] + df["CO2_pure_ice_pebble"]
    
    co2_matrix_total = (
        df["CO2_pure_ice_small"] + df["CO2_pure_ice_pebble"]
        + df["CO2_at_H2O_ice_small"] + df["CO2_at_H2O_ice_pebble"]
    )
    
    h2o_matrix = df["H2O_ice_small"] + df["H2O_ice_pebble"]
    
    # Mass ratios
    co_per_co2_mass = co_at_co2 / np.maximum(co2_matrix_total, floor)
    co_per_h2o_mass = co_at_h2o / np.maximum(h2o_matrix, floor)
    co2_per_h2o_mass = co2_at_h2o / np.maximum(h2o_matrix, floor)
    
    # Molecular guest/host ratios
    co_per_co2_mol = (co_at_co2 / 28.0) / np.maximum(co2_matrix_total / 44.0, floor)
    co_per_h2o_mol = (co_at_h2o / 28.0) / np.maximum(h2o_matrix / 18.0, floor)
    co2_per_h2o_mol = (co2_at_h2o / 44.0) / np.maximum(h2o_matrix / 18.0, floor)

    mask_co2 = co2_matrix_total > 1e-8 * np.nanmax(co2_matrix_total)
    mask_h2o = h2o_matrix > 1e-8 * np.nanmax(h2o_matrix)
    
    plt.figure(figsize=(9, 5))
    plt.semilogx(r, np.where(mask_co2, co_per_co2_mass, np.nan), label="CO@CO2 / CO2, mass", color=color_list[0])
    plt.semilogx(r, np.where(mask_h2o, co_per_h2o_mass, np.nan), label="CO@H2O / H2O, mass", color=color_list[1])
    plt.semilogx(r, np.where(mask_h2o, co2_per_h2o_mass, np.nan), label="CO2@H2O / H2O, mass", color=color_list[2])
    plt.axhline(1.0, color="k", lw=1, ls="--")
    plt.xlabel("Radius [au]")
    plt.ylabel("Guest / host mass ratio")
    plt.title(f"CO trapping capacity, snapshot {snap_index}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/co_trapping_capacity_mass_ratio.png")


    plt.figure(figsize=(9, 5))
    plt.semilogx(r, np.where(mask_co2, co_per_co2_mol, np.nan), label="CO@CO2 / CO2, mol", color=color_list[0])
    plt.semilogx(r, np.where(mask_h2o, co_per_h2o_mol, np.nan), label="CO@H2O / H2O, mol", color=color_list[1])
    plt.semilogx(r, np.where(mask_h2o, co2_per_h2o_mol, np.nan), label="CO2@H2O / H2O, mol", color=color_list[2])
    plt.axhline(1.0, color="k", lw=1, ls="--")
    plt.xlabel("Radius [au]")
    plt.ylabel("Guest / host mass ratio")
    plt.title(f"CO trapping capacity, snapshot {snap_index}")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/co_trapping_capacity_mol_ratio.png")


# -----------------------------
# Time-radius plots from CSV snapshots
# -----------------------------
def pcolormesh_time_radius_computed(
    snapshots,
    value_column,
    outpath,
    title,
    label,
    log_value=False,
    cmap="magma",
):
    frames = []
    times = []

    for snap in snapshots:
        df = read_snapshot(snap.path)
        df = add_volatile_budget_columns_1d(df)

        if value_column not in df.columns:
            continue

        frames.append(df)
        times.append(float(df["time_yr"].iloc[0]))

    if len(frames) < 2:
        return

    r = frames[0]["r_au"].to_numpy(dtype=float)
    values = np.vstack([f[value_column].to_numpy(dtype=float) for f in frames])
    t = np.asarray(times, dtype=float)

    if log_value:
        positive = values[np.isfinite(values) & (values > 0)]
        floor = positive.min() * 1e-3 if positive.size else 1e-300
        plot_values = np.log10(np.maximum(values, floor))
        cb_label = r"$\log_{10}$ " + label
    else:
        plot_values = values
        cb_label = label

    plt.figure(figsize=(9, 5))
    mesh = plt.pcolormesh(r, t, plot_values, shading="auto", cmap=cmap)
    plt.xscale("log")
    plt.xlabel("Radius [au]")
    plt.ylabel("Time [yr]")
    plt.title(title)
    cb = plt.colorbar(mesh)
    cb.set_label(cb_label)
    savefig(outpath)

def make_time_radius_plots(snapshots: Sequence[SnapshotInfo], analysis_dir: Path) -> None:
    # ------------------------------------------------------------
    # Standard time-radius budget plots
    # ------------------------------------------------------------
    targets = [
        (
            "co_hidden_fraction_budget",
            "time_radius_hidden_co_fraction.png",
            "Hidden CO fraction",
            "hidden CO fraction",
            False,
        ),
        (
            "co_gas_fraction_budget",
            "time_radius_gas_co_fraction.png",
            "Gas-phase CO fraction",
            "gas CO fraction",
            False,
        ),
        (
            "co_pebble_fraction_budget",
            "time_radius_co_pebble_fraction.png",
            "Pebble-carried CO fraction",
            "pebble CO fraction",
            False,
        ),
        (
            "co_small_fraction_budget",
            "time_radius_co_small_fraction.png",
            "Small-grain-carried CO fraction",
            "small-grain CO fraction",
            False,
        ),
        (
            "pebble_volatile_ice_budget",
            "time_radius_pebble_volatile_ice.png",
            "Pebble volatile ice surface density",
            r"pebble volatile ice [g cm$^{-2}$]",
            True,
        ),
        (
            "small_volatile_ice_budget",
            "time_radius_small_volatile_ice.png",
            "Small-grain volatile ice surface density",
            r"small-grain volatile ice [g cm$^{-2}$]",
            True,
        ),
        (
            "pebble_fraction_of_solid_volatile_ice",
            "time_radius_pebble_fraction_solid_volatile.png",
            "Pebble fraction of solid volatile ice",
            "pebble fraction",
            False,
        ),
        (
            "small_fraction_of_solid_volatile_ice",
            "time_radius_small_fraction_solid_volatile.png",
            "Small-grain fraction of solid volatile ice",
            "small-grain fraction",
            False,
        ),
        (
            "C_over_O_pebble_masked",
            "time_radius_pebble_c_o_masked.png",
            "Pebble volatile C/O",
            "pebble C/O",
            False,
        ),
        (
            "C_over_O_small_masked",
            "time_radius_small_c_o_masked.png",
            "Small-grain volatile C/O",
            "small-grain C/O",
            False,
        ),
        (
            "co_gas_budget",
            "time_radius_co_gas_surface_density.png",
            "CO gas surface density",
            r"CO gas [g cm$^{-2}$]",
            True,
        ),
        (
            "co_hidden_total_budget",
            "time_radius_hidden_co_surface_density.png",
            "Hidden CO surface density",
            r"hidden CO [g cm$^{-2}$]",
            True,
        ),
    ]

    for col, fname, title, label, logval in targets:
        pcolormesh_time_radius_computed(
            snapshots,
            col,
            analysis_dir / fname,
            title,
            label,
            logval,
        )

    # ------------------------------------------------------------
    # Instantaneous release-rate time-radius plots
    # These use Mdot_dlnr_* and are rates.
    # ------------------------------------------------------------
    release_rate_targets = [
        (
            "Mdot_dlnr_CO_pure",
            "time_radius_Mdot_dlnr_CO_pure.png",
            "Pure CO release rate",
            r"$d\dot{M}_{\rm CO,pure}/d\ln r$ [g s$^{-1}$]",
        ),
        (
            "Mdot_dlnr_CO_at_CO2",
            "time_radius_Mdot_dlnr_CO_at_CO2.png",
            "CO@CO2 release rate",
            r"$d\dot{M}_{\rm CO@CO_2}/d\ln r$ [g s$^{-1}$]",
        ),
        (
            "Mdot_dlnr_CO_at_H2O",
            "time_radius_Mdot_dlnr_CO_at_H2O.png",
            "CO@H2O release rate",
            r"$d\dot{M}_{\rm CO@H_2O}/d\ln r$ [g s$^{-1}$]",
        ),
    ]

    for col, fname, title, label in release_rate_targets:
        # These are positive rates, so log is useful.
        pcolormesh_time_radius_computed(
            snapshots,
            col,
            analysis_dir / fname,
            title,
            label,
            log_value=True,
        )

    # ------------------------------------------------------------
    # Cumulative integrated release profiles
    # Use dM_* rather than summing Mdot_*.
    # ------------------------------------------------------------
    df0 = read_snapshot(snapshots[0].path)
    r = df0["r_au"].to_numpy(dtype=float)
    dlnr = _dlnr_from_centers(r)

    release_channels = [
        ("dM_CO_pure", "CO pure"),
        ("dM_CO_at_CO2", "CO@CO2"),
        ("dM_CO_at_H2O", "CO@H2O"),
    ]

    plt.figure(figsize=(9, 5))

    i = 0
    this_color_list = color_list[:]
    for col, label in release_channels:
        cumulative = np.zeros_like(r)

        for snap in snapshots:
            df = read_snapshot(snap.path)
            if col not in df.columns:
                continue

            # dM is per radial bin. Divide by dlnr for dM/dlnr.
            cumulative += df[col].to_numpy(dtype=float)
        this_color_list = get_color_list(this_color_list, i)
        plt.semilogx(r, cumulative / dlnr / MEARTH, label=label,color=color_list[i])
        i += 1

    plt.xlabel("Radius [au]")
    plt.ylabel(r"$dM_{\rm CO,release}^{\rm cum}/d\ln r$ [M$_\oplus$]")
    plt.title("Cumulative CO release by reservoir")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "total_co_release_dM_dlnr.png")

    plt.figure(figsize=(9, 5))
    i = 0
    this_color_list = color_list[:]
    for col, label in release_channels:
        cumulative = np.zeros_like(r)

        for snap in snapshots:
            df = read_snapshot(snap.path)
            if col not in df.columns:
                continue

            # dM is per radial bin. Divide by dlnr for dM/dlnr.
            cumulative += df[col].to_numpy(dtype=float)
        this_color_list = get_color_list(this_color_list, i)
        plt.semilogx(r, cumulative / MEARTH, label=label, color=this_color_list[i])
        i += 1

    plt.xlabel("Radius [au]")
    plt.ylabel(r"$dM_{\rm CO,release}^{\rm cum}$ [M$_\oplus$]")
    plt.title("Cumulative CO release by reservoir")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "total_co_release_dM.png")

    # ------------------------------------------------------------
    # Cumulative net gas-phase CO change
    # This can be positive or negative.
    # ------------------------------------------------------------
    if "dM_CO_gas" not in df0.columns:
        pass
    cumulative_gas = np.zeros_like(r)

    for snap in snapshots:
        df = read_snapshot(snap.path)
        if "dM_CO_gas" not in df.columns:
            continue
        cumulative_gas += df["dM_CO_gas"].to_numpy(dtype=float)

    plt.figure(figsize=(9, 5))
    
    plt.semilogx(r, cumulative_gas / dlnr / MEARTH, label="net CO gas change", color=color_list[0])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Radius [au]")
    plt.ylabel(r"$dM_{\rm CO,gas}^{\rm net,cum}/d\ln r$ [M$_\oplus$]")
    plt.title("Cumulative net gas-phase CO change")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "total_co_gas_change_dM_dlnr.png")

    plt.figure(figsize=(9, 5))
    plt.semilogx(r, cumulative_gas / MEARTH, label="net CO gas change", color=color_list[0])
    plt.axhline(0.0, linewidth=1)
    plt.xlabel("Radius [au]")
    plt.ylabel(r"$dM_{\rm CO,gas}^{\rm net,cum}$ [M$_\oplus$]")
    plt.title("Cumulative net gas-phase CO change")
    plt.legend(fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "total_co_gas_change_dM.png")


# -----------------------------
# 2D snapshot plots
# -----------------------------
def pcolor_r_z(
    r: np.ndarray,
    z_over_r: np.ndarray,
    values: np.ndarray,
    title: str,
    cb_label: str,
    outpath: Path,
    log_value: bool = False,
    overlay: Optional[Dict[str, np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "magma",
) -> None:
    R = np.broadcast_to(r[:, None], z_over_r.shape)

    values = np.asarray(values, dtype=float)

    if log_value:
        plot_values = np.full_like(values, np.nan, dtype=float)
        good = np.isfinite(values) & (values > 0.0)
        plot_values[good] = np.log10(values[good])
        label = rf"$\log_{{10}}({cb_label})$"
    else:
        plot_values = values
        label = cb_label

    if z_over_r.ndim == 2:
        d0 = np.diff(z_over_r, axis=0)
        d1 = np.diff(z_over_r, axis=1)
        mono0 = np.all(d0 >= 0) or np.all(d0 <= 0)
        mono1 = np.all(d1 >= 0) or np.all(d1 <= 0)
        if not (mono0 and mono1):
            print(f"[pcolor_r_z] non-monotonic z_over_r for: {outpath}")

    plt.figure(figsize=(9, 5))
    mesh = plt.pcolormesh(
        R,
        z_over_r,
        plot_values,
        shading="auto",
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
        rasterized=True,
    )
    plt.xscale("log")
    plt.xlabel("Radius [au]")
    plt.ylabel("z/r")
    plt.title(title)

    cb = plt.colorbar(mesh)
    cb.set_label(label)

    if overlay:
        for i, (name, surf) in enumerate(overlay.items()):
            plt.plot(r, surf, label=name, color=color_list[i])
        plt.legend(fontsize=8, ncols=2)

    savefig(outpath)


def make_2d_plots(data: Dict[str, np.ndarray], analysis_dir: Path, snap_index: int) -> None:
    if not {"r_au", "z_over_r", "T_K"}.issubset(data):
        return

    r = data["r_au"]
    z_over_r = data["z_over_r"]

    overlay = {}
    for key in ["CO_pure", "CO_at_CO2", "CO_at_H2O", "CO2_pure", "CO2_at_H2O", "H2O"]:
        k = f"snow_surface_z_over_r_{key}"
        if k in data:
            overlay[key] = data[k]

    pcolor_r_z(
        r, z_over_r, data["T_K"],
        f"Vertical temperature and snow surfaces, snapshot {snap_index}",
        "T [K]",
        f"{analysis_dir}/selected_2d_temperature_snow_surfaces.png",
        log_value=False,
        overlay=overlay,
        cmap=temp_cmap,
    )

    for key, title in [
        ("survival_CO_pure", "CO pure survival"),
        ("survival_CO_at_CO2", "CO@CO2 survival"),
        ("survival_CO_at_H2O", "CO@H2O survival"),
        ("survival_CO2_pure", "CO2 pure survival"),
        ("survival_CO2_at_H2O", "CO2@H2O survival"),
        ("survival_H2O", "H2O survival"),
    ]:
        if key in data:
            pcolor_r_z(
                r, z_over_r, data[key],
                f"{title}, snapshot {snap_index}",
                "survival probability",
                analysis_dir / f"selected_2d_{key}.png",
                log_value=False,
                overlay=None,
            )

    reservoir_maps = [
        ("surfbin_CO_gas", "CO gas"),
        ("surfbin_CO_pure_ice_pebble", "CO pure ice, pebbles"),
        ("surfbin_CO_at_CO2_ice_pebble", "CO@CO2 ice, pebbles"),
        ("surfbin_CO_at_H2O_ice_pebble", "CO@H2O ice, pebbles"),
        ("surfbin_CO_pure_ice_small", "CO pure ice, small grains"),
        ("surfbin_CO_at_CO2_ice_small", "CO@CO2 ice, small grains"),
        ("surfbin_CO_at_H2O_ice_small", "CO@H2O ice, small grains"),
        ("surfbin_CO2_gas", "CO2 gas"),
        ("surfbin_CO2_pure_ice_pebble", "CO2 pure ice, pebbles"),
        ("surfbin_CO2_at_H2O_ice_pebble", "CO2@H2O ice, pebbles"),
        ("surfbin_CO2_pure_ice_small", "CO2 pure ice, small grains"),
        ("surfbin_CO2_at_H2O_ice_small", "CO2@H2O ice, small grains"),
        ("surfbin_H2O_gas", "H2O gas"),
        ("surfbin_H2O_ice_pebble", "H2O ice, pebbles"),
        ("surfbin_H2O_ice_small", "H2O ice, small grains"),
    ]
    for key, title in reservoir_maps:
        if key in data:
            pcolor_r_z(
                r, z_over_r, data[key],
                f"{title}, snapshot {snap_index}",
                r"vertical-bin surface density [g cm$^{-2}$]",
                Path(f"{analysis_dir}/selected_2d_{key}.png"),
                log_value=True,
                overlay=None,
                vmin=-12, vmax=-2,
            )
               
    hidden_co = (
        data["surfbin_CO_at_CO2_ice_pebble"]
        + data["surfbin_CO_at_H2O_ice_pebble"]
        + data["surfbin_CO_at_CO2_ice_small"]
        + data["surfbin_CO_at_H2O_ice_small"]
    )

    co_tot = (
        data["surfbin_CO_gas"]
        + data["surfbin_CO_pure_ice_pebble"]
        + data["surfbin_CO_pure_ice_small"]
        + data["surfbin_CO_at_CO2_ice_pebble"]
        + data["surfbin_CO_at_H2O_ice_pebble"]
        + data["surfbin_CO_at_CO2_ice_small"]
        + data["surfbin_CO_at_H2O_ice_small"]
    )

    co_floor = 1.0e-12 * np.nanmax(co_tot)

    hidden_frac = np.full_like(co_tot, np.nan, dtype=float)
    np.divide(
        hidden_co,
        co_tot,
        out=hidden_frac,
        where=co_tot > co_floor,
    )

    hidden_frac = np.clip(hidden_frac, 0.0, 1.0)
               
    pcolor_r_z(
        r, z_over_r, data["surfbin_CO_gas"] / co_tot,
        f"CO gas fraction, snapshot {snap_index}",
        r"$\Sigma_{\rm{CO,\, gas}}/\Sigma_{\rm CO}$",
        Path(f"{analysis_dir}/selected_2d_co_gas_frac.png"),
        log_value=True,
        overlay=None,
    )
    # pcolor_r_z(
    #     r, z_over_r, hidden_frac,
    #     f"Hidden CO fraction, snapshot {snap_index}",
    #     r"$\Sigma_{\rm{CO,\, hidden}}/\Sigma_{\rm CO}$",
    #     Path(f"{analysis_dir}/selected_2d_co_hidden_frac.png"),
    #     log_value=True,
    #     overlay=None,
    #     vmin=-6, vmax=0,
    # )
    pcolor_r_z(
        r,
        z_over_r,
        hidden_frac,
        f"Hidden CO fraction, snapshot {snap_index}",
        r"$\Sigma_{\rm CO,hidden}/\Sigma_{\rm CO}$",
        Path(f"{analysis_dir}/selected_2d_co_hidden_frac.png"),
        log_value=False,
        overlay=None,
        vmin=0.0,
        vmax=1.0,
        cmap="viridis",
    )


    bud = compute_volatile_budgets(data, collapse_vertical=False)
    
    co_tot_2d = bud["co_total"]
    small_volatile_2d = bud["small_volatile_ice"]
    pebble_volatile_2d = bud["pebble_volatile_ice"]
    small_c_o_2d = bud["C_over_O_small"]
    pebble_c_o_2d = bud["C_over_O_pebble"]

    z_over_r = data["z_over_r"]
    R = np.broadcast_to(r[:, None], z_over_r.shape)
    
    def plot_2d_quantity(quantity, title, cbar_label, savestr, log=False, vmin=None, vmax=None):
        q = np.asarray(quantity, dtype=float)
    
        if log:
            positive = q[np.isfinite(q) & (q > 0)]
            floor = positive.min() * 1e-3 if positive.size else 1e-300
            qplot = np.log10(np.maximum(q, floor))
            cbar_label = r"$\log_{10}$ " + cbar_label
        else:
            qplot = q
    
        fig, ax = plt.subplots(figsize=(8, 5))
        mesh = ax.pcolormesh(R, z_over_r, qplot, shading="auto", vmin=vmin, vmax=vmax)
        ax.set_xscale("log")
        ax.set_xlabel("Radius [au]")
        ax.set_ylabel("z/r")
        ax.set_title(title)
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(cbar_label)
        plt.tight_layout()
        savefig(savestr)

    plot_2d_quantity(
        small_volatile_2d,
        "Small-grain volatile ice",
        r"[g cm$^{-2}$ per vertical bin]",
        f"{analysis_dir}/small_ice",
        log=True,
    )
    
    plot_2d_quantity(
        pebble_volatile_2d,
        "Pebble volatile ice",
        r"[g cm$^{-2}$ per vertical bin]",
        f"{analysis_dir}/pebble_ice",
        log=True,
    )

    small_mask_2d = small_volatile_2d > 1e-8 * np.nanmax(small_volatile_2d)
    pebble_mask_2d = pebble_volatile_2d > 1e-8 * np.nanmax(pebble_volatile_2d)
    
    small_c_o_2d_masked = np.where(small_mask_2d, small_c_o_2d, np.nan)
    pebble_c_o_2d_masked = np.where(pebble_mask_2d, pebble_c_o_2d, np.nan)
    
    plot_2d_quantity(
        small_c_o_2d_masked,
        "Small-grain volatile C/O",
        "C/O",
        f"{analysis_dir}/small_c_o",
        log=False,
        vmin=0,
        vmax=1,
    )
    
    plot_2d_quantity(
        pebble_c_o_2d_masked,
        "Pebble volatile C/O",
        "C/O",
        f"{analysis_dir}/pebble_c_o",
        log=False,
        vmin=0,
        vmax=1,
    )

    bud_r = compute_volatile_budgets(data, collapse_vertical=True)
    
    r = data["r_au"]
    
    Sigma_small_volatile = bud_r["small_volatile_ice"]
    Sigma_pebble_volatile = bud_r["pebble_volatile_ice"]
    small_c_o = bud_r["C_over_O_small"]
    pebble_c_o = bud_r["C_over_O_pebble"]

    small_mask = bud_r["small_volatile_ice"] > 1e-8 * np.nanmax(bud_r["small_volatile_ice"])
    pebble_mask = bud_r["pebble_volatile_ice"] > 1e-8 * np.nanmax(bud_r["pebble_volatile_ice"])
    
    small_c_o_masked = np.where(small_mask, bud_r["C_over_O_small"], np.nan)
    pebble_c_o_masked = np.where(pebble_mask, bud_r["C_over_O_pebble"], np.nan)


    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.loglog(r, Sigma_pebble_volatile, label="Pebble volatile ice", c=color_list[0])
    ax.loglog(r, Sigma_small_volatile, label="Small-grain volatile ice", c=color_list[1])
    
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel(r"Volatile ice surface density [g cm$^{-2}$]")
    ax.set_title("Carrier-resolved volatile ice budget")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    savefig(f"{analysis_dir}/size_sigma.png")

    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(r, pebble_c_o_masked, label="Pebble C/O", c=color_list[0])
    ax.semilogx(r, small_c_o_masked, label="Small-grain C/O", c=color_list[1])
    
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel("C/O")
    ax.set_title("Carrier-resolved volatile C/O")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    savefig(f"{analysis_dir}/size_c_o.png")

    solid_volatile = Sigma_pebble_volatile + Sigma_small_volatile
    
    pebble_frac = Sigma_pebble_volatile / np.maximum(solid_volatile, 1e-300)
    small_frac = Sigma_small_volatile / np.maximum(solid_volatile, 1e-300)
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(r, pebble_frac, label="Pebble fraction", c=color_list[0])
    ax.semilogx(r, small_frac, label="Small-grain fraction", c=color_list[1])
    
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel("Fraction of solid volatile ice")
    ax.set_title("Which carrier holds the volatile ice?")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    savefig(f"{analysis_dir}/size_frac.png")

    co_gas_frac = bud_r["co_gas_fraction"]
    co_hidden_frac = bud_r["co_hidden_fraction"]
    co_pebble_frac = bud_r["co_pebble_fraction"]
    co_small_frac = bud_r["co_small_fraction"]
    
    fig, ax = plt.subplots(figsize=(8, 5))
    
    ax.semilogx(r, co_gas_frac, label="CO gas / total CO", c=color_list[0])
    ax.semilogx(r, co_hidden_frac, label="Hidden CO / total CO", c=color_list[1])
    ax.semilogx(r, co_pebble_frac, label="Pebble CO / total CO", c=color_list[2])
    ax.semilogx(r, co_small_frac, label="Small-grain CO / total CO", c=color_list[3])
    
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel("Fraction")
    ax.set_title("CO partitioning")
    ax.set_ylim(-0.02, 1.02)
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    savefig(f"{analysis_dir}/co_partition.png")


# -----------------------------
# Summary metrics
# -----------------------------
def write_summary_metrics(
    df: pd.DataFrame,
    diag: Optional[pd.DataFrame],
    outpath: Path,
    snap_index: int,
) -> None:
    rows = []

    def add(name: str, value: float, desc: str) -> None:
        rows.append({"metric": name, "value": value, "description": desc})

    if "time_yr" in df.columns:
        add("snapshot_index", float(snap_index), "Selected snapshot index.")
        add("snapshot_time_yr", float(df["time_yr"].iloc[0]), "Selected snapshot time.")

    if has_columns(df, ["r_au", "hidden_CO_fraction"]):
        vals = df["hidden_CO_fraction"].to_numpy()
        j = int(np.nanargmax(vals))
        add("max_hidden_CO_fraction", float(np.nanmax(vals)), "Maximum radial hidden-CO fraction.")
        add("r_at_max_hidden_CO_fraction_au", float(df["r_au"].iloc[j]), "Radius of max hidden-CO fraction.")

    if has_columns(df, ["gas_CO_fraction"]):
        add("min_gas_CO_fraction", float(np.nanmin(df["gas_CO_fraction"])), "Minimum radial gas-phase CO fraction.")

    if has_columns(df, ["C_over_O_pebble"]):
        vals = df["C_over_O_pebble"].replace([np.inf, -np.inf], np.nan).to_numpy()
        add("median_pebble_C_over_O", float(np.nanmedian(vals)), "Median pebble C/O over radial grid.")

    if diag is not None and not diag.empty:
        for col in [
            "M_CO_total_mearth", "M_CO_gas_mearth", "M_CO_hidden_mearth", "M_CO_solid_mearth",
            "M_CO2_total_mearth", "M_H2O_total_mearth",
        ]:
            if col in diag.columns:
                add(f"final_{col}", float(diag[col].iloc[-1]), f"Final diagnostic value: {col}.")
        if has_columns(diag, ["M_CO_total_mearth", "M_CO_hidden_mearth"]):
            total = max(float(diag["M_CO_total_mearth"].iloc[-1]), EPS)
            hidden = float(diag["M_CO_hidden_mearth"].iloc[-1])
            add("final_global_hidden_CO_fraction", hidden / total, "Final global hidden-CO fraction.")

    with open(outpath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["metric", "value", "description"])
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze mixed_ice_transport_1p1d outputs.")
    parser.add_argument("output_dir", type=str, help="Run output directory from mixed_ice_transport_1p1d.py.")
    parser.add_argument("--analysis-dir", type=str, default=None, help="Plot output directory. Default: <output_dir>/analysis_plots")
    parser.add_argument("--snap", type=str, default="latest", help="Snapshot: latest, first, middle, or integer index.")
    parser.add_argument("--max-time-radius-snaps", type=int, default=200, help="Maximum snapshots used for time-radius plots.")
    parser.add_argument("--skip-2d", action="store_true", help="Skip 2D npz plots.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    analysis_dir = Path(args.analysis_dir).expanduser().resolve() if args.analysis_dir else output_dir / "analysis_plots"
    ensure_dir(analysis_dir)

    snapshots = list_snapshots(output_dir)
    if not snapshots:
        raise FileNotFoundError(f"No 1D snapshot CSV files found in {output_dir / 'snapshots'}")

    selected = select_snapshot(snapshots, args.snap)
    selected_df = read_snapshot(selected.path)
    diag = read_diagnostics(output_dir)

    make_diagnostics_plots(diag, analysis_dir)
    make_final_1d_plots(selected_df, analysis_dir, selected.index)

    if len(snapshots) > args.max_time_radius_snaps:
        indices = np.linspace(0, len(snapshots) - 1, args.max_time_radius_snaps).astype(int)
        tr_snaps = [snapshots[i] for i in np.unique(indices)]
    else:
        tr_snaps = snapshots
    make_time_radius_plots(tr_snaps, analysis_dir)

    if not args.skip_2d:
        snaps2d = list_snapshots_2d(output_dir)
        if snaps2d:
            nearest = min(snaps2d, key=lambda s: abs(s.index - selected.index))
            data2d = read_2d_snapshot(nearest.path)
            make_2d_plots(data2d, analysis_dir, nearest.index)

    write_summary_metrics(selected_df, diag, analysis_dir / "summary_metrics.csv", selected.index)
    print(f"Analysis complete. Plots written to: {analysis_dir}")
    
    
if __name__ == "__main__":
    main()