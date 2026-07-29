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
    diagnostics_boundary_losses.png
    diagnostics_phase_exchange.png
    diagnostics_capacity_limiting.png
    diagnostics_numerical_quality.png
    diagnostics_timestep.png
    diagnostics_backreaction_strength.png  [when applicable]
    capacity_rejection_profile.png
    co_phase_exchange_profile.png
    final_volatile_profiles.png
    final_carrier_profiles.png
    final_c_o_profiles.png
    final_snow_surfaces.png
    time_radius_*.png
    selected_2d_*.png
    paper_radial_co_and_carrier_composition.png
    paper_cumulative_release_profile.png
    paper_fiducial_morphology.png
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


try:
    from colorspacious import cspace_converter
except ImportError:  # Optional plotting enhancement.
    cspace_converter = None
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib import rc as mplrc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm, SymLogNorm, Normalize


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
    if cspace_converter is None:
        rgb_colors = np.linspace(start_color, end_color, N)
    else:
        converter = cspace_converter("sRGB1", "CAM02-UCS")
        start_color_lab = converter(start_color)
        end_color_lab = converter(end_color)
        lab_colors = np.linspace(start_color_lab, end_color_lab, N)
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

c0 = [0.2745098 , 0.4       , 0.55294118]
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
              [0.98431373, 0.62745098, 0.40784314],
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
    """Plot global inventories plus runtime-only transport/phase diagnostics."""
    if diag is None or diag.empty or "time_yr" not in diag.columns:
        return
    t = diag["time_yr"].to_numpy(dtype=float)

    mass_cols = [
        "M_CO_total_mearth", "M_CO_gas_mearth", "M_CO_solid_mearth", "M_CO_hidden_mearth",
        "M_CO2_total_mearth", "M_CO2_gas_mearth", "M_CO2_solid_mearth",
        "M_H2O_total_mearth", "M_H2O_gas_mearth", "M_H2O_solid_mearth",
    ]
    mass_labels = ["CO total", "CO gas", "CO solid", "CO matrix", "CO2 total", "CO2 gas", "CO2 solid", "H2O total", "H2O gas", "H2O solid"]
    plt.figure(figsize=(9, 5))
    this_colors = color_list[:]
    for i, (colname, label) in enumerate(zip(mass_cols, mass_labels)):
        if colname in diag.columns:
            this_colors = get_color_list(this_colors, i)
            plt.plot(t, np.maximum(diag[colname], EPS), label=label, color=this_colors[i])
    plt.yscale("log"); plt.xlabel("Time [yr]"); plt.ylabel("Mass [Earth masses]"); plt.title("Global volatile inventories")
    plt.legend(ncols=2, fontsize=8); plt.grid(True, which="both", alpha=0.3)
    savefig(Path(analysis_dir) / "diagnostics_masses.png")

    if has_columns(diag, ["M_CO_total_mearth", "M_CO_gas_mearth", "M_CO_hidden_mearth"]):
        total = np.maximum(diag["M_CO_total_mearth"].to_numpy(), EPS)
        plt.figure(figsize=(8, 5))
        plt.plot(t, diag["M_CO_gas_mearth"] / total, label="CO gas / total CO", color=color_list[0])
        plt.plot(t, diag["M_CO_hidden_mearth"] / total, label="matrix-associated CO / total CO", color=color_list[1])
        if "M_CO_solid_mearth" in diag.columns:
            plt.plot(t, diag["M_CO_solid_mearth"] / total, label="solid CO / total CO", color=color_list[2])
        plt.xlabel("Time [yr]"); plt.ylabel("Fraction"); plt.ylim(-0.02, 1.02); plt.title("Global CO partitioning")
        plt.legend(); plt.grid(True, alpha=0.3)
        savefig(Path(analysis_dir) / "diagnostics_fractions.png")

    # Cumulative boundary losses distinguish inward delivery from outer-domain loss.
    boundary_cols = [f"cum_boundary_{side}_{sp}_mearth" for side in ("inner", "outer") for sp in ("CO", "CO2", "H2O")]
    if any(c in diag.columns for c in boundary_cols):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
        for ax, side, title in zip(axes, ("inner", "outer"), ("Inner-boundary delivery", "Outer-boundary loss")):
            for i, sp in enumerate(("CO", "CO2", "H2O")):
                c = f"cum_boundary_{side}_{sp}_mearth"
                if c in diag.columns:
                    ax.plot(t, diag[c], label=sp, color=color_list[i])
            ax.set_title(title); ax.set_xlabel("Time [yr]"); ax.grid(True, alpha=0.25); ax.legend()
        axes[0].set_ylabel("Cumulative mass [Earth masses]")
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "diagnostics_boundary_losses.png", dpi=220); plt.close(fig)

    # Sign-separated gas phase exchange shows gross cycling versus net production.
    if any(f"cum_phase_gas_gain_{sp}_mearth" in diag.columns for sp in ("CO", "CO2", "H2O")):
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), sharex=True)
        for ax, sp in zip(axes, ("CO", "CO2", "H2O")):
            for suffix, label, ls in (("gain", "positive gas gain", "-"), ("loss", "gas loss to solids", "--"), ("net", "net phase source", ":")):
                c = f"cum_phase_gas_{suffix}_{sp}_mearth"
                if c in diag.columns:
                    ax.plot(t, diag[c], label=label, linestyle=ls)
            ax.axhline(0.0, color="0.5", linewidth=0.8); ax.set_title(sp); ax.set_xlabel("Time [yr]"); ax.grid(True, alpha=0.25)
        axes[0].set_ylabel("Cumulative phase exchange [Earth masses]"); axes[-1].legend(fontsize=8)
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "diagnostics_phase_exchange.png", dpi=220); plt.close(fig)

    # Host-capacity diagnostics: instantaneous target rejection and gross throughput.
    cap_channels = ("CO_at_CO2", "CO_at_H2O", "CO2_at_H2O")
    if any(f"total_capacity_excess_{ch}_mearth" in diag.columns for ch in cap_channels):
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
        labels = {"CO_at_CO2": r"CO@CO$_2$", "CO_at_H2O": r"CO@H$_2$O", "CO2_at_H2O": r"CO$_2$@H$_2$O"}
        for i, ch in enumerate(cap_channels):
            ccur = f"current_capacity_excess_{ch}_mearth"
            ctot = f"total_capacity_excess_{ch}_mearth"
            if ccur in diag.columns: axes[0].plot(t, diag[ccur], label=labels[ch], color=color_list[i])
            if ctot in diag.columns: axes[1].plot(t, diag[ctot], label=labels[ch], color=color_list[i])
        for i, sp in enumerate(("CO", "CO2")):
            c = f"total_capacity_to_gas_{sp}_mearth"
            if c in diag.columns: axes[2].plot(t, diag[c], label=sp, color=color_list[i])
        axes[0].set_title("Current target rejected by host caps"); axes[0].set_ylabel("Instantaneous target excess [Earth masses]")
        axes[1].set_title("Gross guest rejected by host caps"); axes[1].set_ylabel("Cumulative gross throughput [Earth masses]")
        axes[2].set_title("Capacity-rejected material sent to gas"); axes[2].set_ylabel("Cumulative gross throughput [Earth masses]")
        for ax in axes:
            ax.set_xlabel("Time [yr]"); ax.grid(True, alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "diagnostics_capacity_limiting.png", dpi=220); plt.close(fig)

    # Numerical mass balance and positivity corrections.
    residual_cols = [f"mass_balance_residual_fraction_{sp}" for sp in ("CO", "CO2", "H2O")]
    clip_cols = [f"cum_clipped_added_{sp}_mearth" for sp in ("CO", "CO2", "H2O")]
    if any(c in diag.columns for c in residual_cols + clip_cols):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        for i, sp in enumerate(("CO", "CO2", "H2O")):
            rc = f"mass_balance_residual_fraction_{sp}"
            cc = f"cum_clipped_added_{sp}_mearth"
            if rc in diag.columns: axes[0].plot(t, diag[rc], label=sp, color=color_list[i])
            if cc in diag.columns: axes[1].plot(t, np.maximum(diag[cc], 1e-30), label=sp, color=color_list[i])
        axes[0].axhline(0.0, color="0.5", linewidth=0.8); axes[0].set_title("Species mass-balance residual"); axes[0].set_ylabel(r"$(M+M_{out}-M_0-M_{clip})/M_0$")
        axes[1].set_yscale("log"); axes[1].set_title("Cumulative positivity correction"); axes[1].set_ylabel("Added mass [Earth masses]")
        for ax in axes: ax.set_xlabel("Time [yr]"); ax.grid(True, which="both", alpha=0.25); ax.legend()
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "diagnostics_numerical_quality.png", dpi=220); plt.close(fig)

    if has_columns(diag, ["dt_min_yr", "dt_mean_yr", "dt_max_yr"]):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        axes[0].plot(t, diag["dt_min_yr"], label="minimum"); axes[0].plot(t, diag["dt_mean_yr"], label="mean"); axes[0].plot(t, diag["dt_max_yr"], label="maximum")
        axes[0].set_yscale("log"); axes[0].set_ylabel("Timestep [yr]"); axes[0].legend(); axes[0].set_title("Timestep statistics")
        for c, label in (("n_advective_limited", "advective"), ("n_diffusive_limited", "diffusive"), ("n_max_timestep_limited", "maximum dt"), ("n_output_limited", "output/end")):
            if c in diag.columns: axes[1].plot(t, diag[c], label=label)
        axes[1].set_ylabel("Cumulative step count"); axes[1].set_title("Timestep limiter") ; axes[1].legend(fontsize=8)
        for ax in axes: ax.set_xlabel("Time [yr]"); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "diagnostics_timestep.png", dpi=220); plt.close(fig)

    br_cols = ["mass_weighted_epsilon_pebble", "mass_weighted_rel_delta_v_gas_backreaction", "mass_weighted_rel_delta_v_pebble_backreaction"]
    if any(c in diag.columns and np.nanmax(np.abs(diag[c])) > 0 for c in br_cols):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
        if "mass_weighted_epsilon_pebble" in diag.columns:
            axes[0].plot(t, diag["mass_weighted_epsilon_pebble"], label="mass-weighted"); axes[0].plot(t, diag.get("max_epsilon_pebble", 0), label="maximum", linestyle="--")
        axes[0].set_title("Pebble loading"); axes[0].set_ylabel(r"$\epsilon_{\rm peb}$"); axes[0].legend()
        for c, label in (("mass_weighted_rel_delta_v_gas_backreaction", "gas"), ("mass_weighted_rel_delta_v_pebble_backreaction", "pebbles")):
            if c in diag.columns: axes[1].plot(t, diag[c], label=label)
        axes[1].set_title("Backreaction velocity modification"); axes[1].set_ylabel(r"mass-weighted $|v-v_0|/|v_0|$"); axes[1].legend()
        for ax in axes: ax.set_xlabel("Time [yr]"); ax.grid(True, alpha=0.25)
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "diagnostics_backreaction_strength.png", dpi=220); plt.close(fig)

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
def make_final_1d_plots(df: pd.DataFrame, analysis_dir: Path, snap_index: int,
                        plot_entrap_surface: bool = True, plot_gas: bool = False) -> None:
    if "r_au" not in df.columns:
        return
    r = df["r_au"]

    plt.figure(figsize=(9, 5))
    i = 0
    this_color_list = color_list[:]
    for col, label in [
        ("CO_gas", "CO gas"), ("CO_solid_total", "CO solid"), ("CO_hidden_total", "Hidden CO"),
        ("CO2_gas", r"CO$_{2}$ gas"), ("CO2_solid_total", r"CO$_{2}$ solid"),
        ("H2O_gas", r"H$_{2}$O gas"), ("H2O_solid_total", r"H$_{2}$O solid"),
    ]:
        if col in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.loglog(r, np.maximum(df[col], EPS), label=label, color=this_color_list[i])
            i += 1
    plt.xlabel("Radius [au]")
    plt.ylabel(r"Surface density [g cm$^{-2}$]")
    plt.title(f"Volatile reservoir profiles")#, snapshot {snap_index}")
    plt.ylim(bottom=1e-8)
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "final_volatile_profiles.png")

    plt.figure(figsize=(9, 5))
    this_color_list = color_list[:]
    # for col, label, i, ls in [
    #     ("CO_solid_pebble", "CO in pebbles", 0, '-'), ("CO_solid_small", "CO in small grains", 0, '--'),
    #     ("CO2_solid_pebble", r"CO$_{2}$ in pebbles", 1, '-'), ("CO2_solid_small", r"CO$_{2}$ in small grains", 1, '--'),
    #     ("H2O_solid_pebble", r"H$_{2}$O in pebbles", 2, '-'), ("H2O_solid_small", r"H$_{2}$O in small grains", 2, '--'),
    #     ("ref_solid_pebble", r"Refractory pebbles", 3, '-'), ("ref_solid_small", r"Refractory small grains", 3, '--'),
    # ]:
    for col, label, i, ls in [
        ("CO_solid_pebble", "CO", 0, '-'), ("CO_solid_small", None, 0, '--'),
        ("CO2_solid_pebble", r"CO$_{2}$", 1, '-'), ("CO2_solid_small", None, 1, '--'),
        ("H2O_solid_pebble", r"H$_{2}$O", 2, '-'), ("H2O_solid_small", None, 2, '--'),
        ("ref_solid_pebble", r"Refractory", 3, '-'), ("ref_solid_small", None, 3, '--'),
        ("CO_gas", None, 0, ':'), ("CO2_gas", None, 1, ':'), ("H2O_gas", None, 2, ':'),
    ]:
        if not plot_gas and "_gas" in col:
            continue
        if col in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.loglog(r, np.maximum(df[col], EPS), label=label, color=this_color_list[i], ls=ls)
    plt.plot([], [], c='k', lw=1, label='Pebbles')
    plt.plot([], [], c='k', lw=1, ls='--', label='Small grains')
    if plot_gas:
        plt.plot([], [], c='k', lw=1, ls=':', label='Gas')
    plt.xlabel("Radius [au]")
    plt.ylabel(r"Surface density [g cm$^{-2}$]")
    plt.ylim(bottom=1e-8)
    plt.title(f"Carrier-resolved solid profiles")#, snapshot {snap_index}")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/final_carrier_profiles.png")

    plt.figure(figsize=(9, 5))
    i = 0
    this_color_list = color_list[:]
    for col, label in [
        ("C_over_O_gas", "gas C/O"),
        ("C_over_O_solid", "solid C/O"),
        ("C_over_O_pebble", "pebble volatile C/O"),
        ("C_over_O_small", "small-grain volatile C/O"),
        ("hidden_CO_fraction", "matrix-associated CO fraction"),
        ("gas_CO_fraction", "gas CO fraction"),
    ]:
        if col in df.columns:
            this_color_list = get_color_list(this_color_list, i)
            plt.semilogx(r, df[col], label=label, color=this_color_list[i])
            i += 1
    plt.xlabel("Radius [au]")
    plt.ylabel("Ratio / fraction")
    plt.title(f"C/O and CO partitioning")#, snapshot {snap_index}")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(f"{analysis_dir}/final_c_o_profiles.png")

    snow_cols = [c for c in df.columns if c.startswith("snow_surface_z_over_r_")]
    if not plot_entrap_surface:
        snow_cols = [c for c in snow_cols if "_at_" not in c]
    if snow_cols:
        i = 0
        this_color_list = color_list[:]
        plt.figure(figsize=(9, 5))
        # breakpoint()
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
        plt.title(f"Modeled release surfaces")#, snapshot {snap_index}")
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
        plt.ylabel(r"$dM_{\rm ice,release}$  [M$_{\oplus}$]")
        plt.title(f"Ice mass release")#, snapshot {snap_index}")
        plt.legend(ncols=2, fontsize=8)
        plt.grid(True, which="both", alpha=0.3)
        savefig(f"{analysis_dir}/ice_release.png")


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
    plt.title(f"CO ices")#, snapshot {snap_index}")
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
    plt.title(f"CO@X/X total")#, snapshot {snap_index}")
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
    plt.title(f"CO trapping capacity")#"snap_index}")
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
    plt.title(f"CO trapping capacity")#, snapshot {snap_index}")
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
    vmin=None, vmax=None,
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
        # plot_values = np.log10(np.maximum(values, floor))
        # cb_label = r"$\log_{10}$ " + label
        plot_values = np.maximum(values, floor)
        cb_label = label
        norm = LogNorm(vmin=vmin, vmax=vmax)
    else:
        plot_values = values
        cb_label = label
        norm = Normalize(vmin=vmin, vmax=vmax)

    plt.figure(figsize=(9, 5))
    mesh = plt.pcolormesh(r, t, plot_values, shading="auto", cmap=cmap, norm=norm, rasterized=True)
    plt.xscale("log")
    plt.xlabel("Radius [au]")
    plt.ylabel("Time [yr]")
    plt.title(title)
    cb = plt.colorbar(mesh)
    cb.set_label(cb_label)
    savefig(outpath)
    
def make_paper_time_radius_co_partitioning(
    snapshots,
    analysis_dir,
    fname="paper_time_radius_co_partitioning.png",
    cmap="magma",
    total_floor_fraction=1.0e-10,
):
    """
    Make a three-panel time-radius figure showing the mutually exclusive
    partition of local CO among:

      (a) gas-phase CO
      (b) pure CO ice
      (c) matrix-associated CO = CO@CO2 + CO@H2O

    At every well-populated radial cell,

        f_gas + f_pure + f_matrix = 1.

    Parameters
    ----------
    snapshots : sequence of SnapshotInfo
        Snapshot list. Each element must have a `.path` attribute.

    analysis_dir : str or pathlib.Path
        Directory in which the figure will be saved.

    fname : str
        Output filename.

    cmap : str or matplotlib colormap
        Colormap used for all panels.

    total_floor_fraction : float
        Cells with total CO below this fraction of the maximum total CO in
        that snapshot are masked. This avoids plotting meaningless fractions
        where effectively no CO remains.
    """

    if snapshots is None or len(snapshots) == 0:
        return

    analysis_dir = Path(analysis_dir)

    def linear_edges_from_centers(x):
        """Construct cell edges from linearly spaced or irregular centers."""
        x = np.asarray(x, dtype=float)

        if x.ndim != 1 or x.size == 0:
            raise ValueError("Centers must be a non-empty 1D array.")

        if x.size == 1:
            dx = 0.5 * max(abs(x[0]), 1.0)
            return np.array([x[0] - dx, x[0] + dx])

        edges = np.empty(x.size + 1, dtype=float)
        edges[1:-1] = 0.5 * (x[:-1] + x[1:])
        edges[0] = x[0] - 0.5 * (x[1] - x[0])
        edges[-1] = x[-1] + 0.5 * (x[-1] - x[-2])

        return edges

    def log_edges_from_centers(x):
        """Construct logarithmic cell edges from positive radial centers."""
        x = np.asarray(x, dtype=float)

        if np.any(x <= 0.0):
            raise ValueError("Radial centers must be positive.")

        return np.exp(linear_edges_from_centers(np.log(x)))


    times = []
    gas_rows = []
    pure_rows = []
    matrix_rows = []

    r_ref = None

    for snap in snapshots:
        df = read_snapshot(snap.path)

        if "r_au" not in df.columns:
            continue

        r_now = df["r_au"].to_numpy(dtype=float)

        if r_ref is None:
            r_ref = r_now
        elif (
            len(r_now) != len(r_ref)
            or not np.allclose(r_now, r_ref, rtol=1.0e-10, atol=0.0)
        ):
            raise ValueError(
                "The snapshot radial grids do not match; "
                "a time-radius map cannot be constructed."
            )

        if "time_yr" in df.columns:
            time_yr = float(df["time_yr"].iloc[0])
        elif hasattr(snap, "time_yr"):
            time_yr = float(snap.time_yr)
        else:
            time_yr = float(len(times))

        # Mutually exclusive CO reservoirs.
        co_gas = _col(df, "CO_gas")

        co_pure = (
            _col(df, "CO_pure_ice_pebble")
            + _col(df, "CO_pure_ice_small")
        )

        co_at_co2 = (
            _col(df, "CO_at_CO2_ice_pebble")
            + _col(df, "CO_at_CO2_ice_small")
        )

        co_at_h2o = (
            _col(df, "CO_at_H2O_ice_pebble")
            + _col(df, "CO_at_H2O_ice_small")
        )

        co_matrix = co_at_co2 + co_at_h2o
        co_total = co_gas + co_pure + co_matrix

        co_floor = (
            total_floor_fraction
            * max(float(np.nanmax(co_total)), EPS)
        )
        good = co_total > co_floor

        gas_fraction = np.full_like(co_total, np.nan, dtype=float)
        pure_fraction = np.full_like(co_total, np.nan, dtype=float)
        matrix_fraction = np.full_like(co_total, np.nan, dtype=float)

        np.divide(
            co_gas,
            co_total,
            out=gas_fraction,
            where=good,
        )
        np.divide(
            co_pure,
            co_total,
            out=pure_fraction,
            where=good,
        )
        np.divide(
            co_matrix,
            co_total,
            out=matrix_fraction,
            where=good,
        )

        gas_fraction = np.clip(gas_fraction, 0.0, 1.0)
        pure_fraction = np.clip(pure_fraction, 0.0, 1.0)
        matrix_fraction = np.clip(matrix_fraction, 0.0, 1.0)

        times.append(time_yr)
        gas_rows.append(gas_fraction)
        pure_rows.append(pure_fraction)
        matrix_rows.append(matrix_fraction)

    if r_ref is None or len(times) < 2:
        return

    times = np.asarray(times, dtype=float)
    gas_map = np.asarray(gas_rows, dtype=float)
    pure_map = np.asarray(pure_rows, dtype=float)
    matrix_map = np.asarray(matrix_rows, dtype=float)

    # Ensure chronological ordering.
    order = np.argsort(times)
    times = times[order]
    gas_map = gas_map[order]
    pure_map = pure_map[order]
    matrix_map = matrix_map[order]

    r_edges = log_edges_from_centers(r_ref)
    time_edges = linear_edges_from_centers(times)

    # Avoid extending the first plotting cell to negative time.
    time_edges[0] = max(0.0, time_edges[0])

    R_edges, T_edges = np.meshgrid(r_edges, time_edges)

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(15.5, 4.8),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    mesh = None

    for ax, (values, title) in zip(
        axes,
        [
            (gas_map, "(a) Gas phase"),
            (pure_map, "(b) Pure CO ice"),
            (matrix_map, "(c) Matrix-associated CO"),
        ],
    ):
        mesh = ax.pcolormesh(
            R_edges,
            T_edges,
            values,
            shading="flat",
            cmap=cmap,
            vmin=0.0,
            vmax=1.0,
            rasterized=True,
        )

        ax.set_xscale("log")
        ax.set_xlabel("Radius [au]")
        ax.set_title(title)
        ax.grid(False)

    axes[0].set_ylabel("Time [yr]")

    cbar = fig.colorbar(
        mesh,
        ax=axes,
        location="right",
        pad=0.015,
        fraction=0.025,
    )
    cbar.set_label("Fraction of local CO")

    fig.suptitle("Time evolution of radial CO partitioning", y=1.03)

    fig.savefig(
        analysis_dir / fname,
        dpi=240,
        bbox_inches="tight",
    )
    plt.close(fig)

def make_time_radius_plots(snapshots: Sequence[SnapshotInfo], analysis_dir: Path) -> None:
    # ------------------------------------------------------------
    # Standard time-radius budget plots
    # ------------------------------------------------------------
    targets = [
        (
            "co_hidden_fraction_budget",
            "time_radius_hidden_co_fraction.png",
            "Matrix-associated CO fraction",
            "matrix-associated CO fraction",
            False,
            None, None,
        ),
        (
            "co_gas_fraction_budget",
            "time_radius_gas_co_fraction.png",
            "Gas-phase CO fraction",
            "gas CO fraction",
            False,
            None, None,
        ),
        (
            "co_pebble_fraction_budget",
            "time_radius_co_pebble_fraction.png",
            "Pebble-carried CO fraction",
            "pebble CO fraction",
            False,
            None, None,
        ),
        (
            "co_small_fraction_budget",
            "time_radius_co_small_fraction.png",
            "Small-grain-carried CO fraction",
            "small-grain CO fraction",
            False,
            None, None,
        ),
        (
            "pebble_volatile_ice_budget",
            "time_radius_pebble_volatile_ice.png",
            "Pebble volatile ice surface density",
            r"pebble volatile ice [g cm$^{-2}$]",
            True,
            None, None,
        ),
        (
            "small_volatile_ice_budget",
            "time_radius_small_volatile_ice.png",
            "Small-grain volatile ice surface density",
            r"small-grain volatile ice [g cm$^{-2}$]",
            True,
            None, None,
        ),
        (
            "pebble_fraction_of_solid_volatile_ice",
            "time_radius_pebble_fraction_solid_volatile.png",
            "Pebble fraction of solid volatile ice",
            "pebble fraction",
            False,
            None, None,
        ),
        (
            "small_fraction_of_solid_volatile_ice",
            "time_radius_small_fraction_solid_volatile.png",
            "Small-grain fraction of solid volatile ice",
            "small-grain fraction",
            False,
            None, None,
        ),
        (
            "C_over_O_pebble_masked",
            "time_radius_pebble_c_o_masked.png",
            "Pebble volatile C/O",
            "pebble volatile C/O",
            True,
            1e-3, 1e1,
        ),
        (
            "C_over_O_small_masked",
            "time_radius_small_c_o_masked.png",
            "Small-grain volatile C/O",
            "small-grain volatile C/O",
            True,
            1e-3, 1e1,
        ),
        (
            "co_gas_budget",
            "time_radius_co_gas_surface_density.png",
            "CO gas surface density",
            r"CO gas [g cm$^{-2}$]",
            True,
            None, None,
        ),
        (
            "co_hidden_total_budget",
            "time_radius_hidden_co_surface_density.png",
            "Hidden CO surface density",
            r"hidden CO [g cm$^{-2}$]",
            True,
            None, None,
        ),
    ]

    for col, fname, title, label, logval, vmin, vmax in targets:
        pcolormesh_time_radius_computed(
            snapshots,
            col,
            analysis_dir / fname,
            title,
            label,
            logval,
            vmin=vmin, vmax=vmax,
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
        ("dM_CO2_pure", "CO2 pure"),
        ("dM_CO2_at_H2O", "CO2@H2O"),
        # ("dM_H2O_pure", "H2O pure"), # maybe add pure
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
    plt.ylabel(r"$dM_{\rm ice,release}^{\rm cum}/d\ln r$ [M$_\oplus$]")
    plt.title("Cumulative gross volatile-ice reservoir loss")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "total_ice_release_dM_dlnr.png")

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
    plt.ylabel(r"$dM_{\rm ice,release}^{\rm cum}$ per radial bin [M$_\oplus$]")
    plt.title("Cumulative gross volatile-ice reservoir loss")
    plt.legend(ncols=2, fontsize=8)
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "cumulative_volatile_ice_reservoir_loss.png")

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
def _edges_from_centers_1d(x: np.ndarray, log: bool = False) -> np.ndarray:
    """Return cell edges from monotonically increasing cell centers."""
    x = np.asarray(x, dtype=float)
    if x.ndim != 1 or x.size < 2:
        raise ValueError("Need at least two 1D centers to infer cell edges.")

    work = np.log(x) if log else x
    edges = np.empty(work.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (work[:-1] + work[1:])
    edges[0] = work[0] - 0.5 * (work[1] - work[0])
    edges[-1] = work[-1] + 0.5 * (work[-1] - work[-2])
    return np.exp(edges) if log else edges


def _rz_cell_edges(r: np.ndarray, z_over_r: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Construct explicit curvilinear cell-edge coordinates for r--z/r plots.

    The old plotting routine passed cell-centered 2D z/r coordinates directly
    to pcolormesh. Because z/r varies with radius, those coordinates need not be
    monotonic in both directions, and Matplotlib can infer incorrect cell edges.
    This function builds the edges explicitly, which removes the common
    pcolormesh warning and avoids small spurious rectangular chunks.
    """
    r = np.asarray(r, dtype=float)
    zc = np.asarray(z_over_r, dtype=float)

    if zc.ndim != 2:
        raise ValueError("z_over_r must be a 2D array with shape (n_r, n_z).")
    if zc.shape[0] != r.size:
        raise ValueError(
            f"Expected z_over_r.shape[0] == len(r), got {zc.shape[0]} and {r.size}."
        )
    if zc.shape[1] < 2:
        raise ValueError("Need at least two vertical centers to infer z/r edges.")

    # Radial edges are log-spaced because the radial grid is log-spaced.
    r_edges = _edges_from_centers_1d(r, log=True)

    # First infer vertical edges at each radial cell center.
    z_vert_edges = np.empty((zc.shape[0], zc.shape[1] + 1), dtype=float)
    z_vert_edges[:, 1:-1] = 0.5 * (zc[:, :-1] + zc[:, 1:])
    z_vert_edges[:, 0] = zc[:, 0] - 0.5 * (zc[:, 1] - zc[:, 0])
    z_vert_edges[:, -1] = zc[:, -1] + 0.5 * (zc[:, -1] - zc[:, -2])

    # Then infer those vertical-edge coordinates at radial cell edges.
    z_edges = np.empty((zc.shape[0] + 1, zc.shape[1] + 1), dtype=float)
    z_edges[1:-1, :] = 0.5 * (z_vert_edges[:-1, :] + z_vert_edges[1:, :])
    z_edges[0, :] = z_vert_edges[0, :] - 0.5 * (z_vert_edges[1, :] - z_vert_edges[0, :])
    z_edges[-1, :] = z_vert_edges[-1, :] + 0.5 * (z_vert_edges[-1, :] - z_vert_edges[-2, :])

    # The plots show the upper half column, so the lower edge should not dip below zero.
    z_edges = np.maximum(z_edges, 0.0)
    z_edges[:, 0] = 0.0

    R_edges = np.broadcast_to(r_edges[:, None], z_edges.shape)
    return R_edges, z_edges


def _prepare_pcolor_values(
    values: np.ndarray,
    log_value: bool,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
) -> Tuple[np.ma.MaskedArray, str]:
    """
    Prepare values for pcolormesh. For log plots, non-positive finite values are
    put at the plotting floor instead of becoming NaN. This prevents zero-valued
    reservoirs from appearing as distracting black/blank chunks.
    """
    values = np.asarray(values, dtype=float)

    if not log_value:
        return np.ma.masked_invalid(values), ""

    positive = values[np.isfinite(values) & (values > 0.0)]
    if vmin is not None:
        # floor = 10.0 ** float(vmin)
        floor = float(vmin)
    elif positive.size:
        floor = max(float(np.nanmin(positive)) * 1.0e-3, 1.0e-300)
    else:
        floor = 1.0e-300

    plot_values = np.full_like(values, np.nan, dtype=float)
    finite = np.isfinite(values)
    # plot_values[finite] = np.log10(np.maximum(values[finite], floor))
    plot_values[finite] = np.maximum(values[finite], floor)
    # return np.ma.masked_invalid(plot_values), r"$\log_{10}$"
    return np.ma.masked_invalid(plot_values), r""


def _copy_cmap_with_bad(cmap):
    cmap_obj = plt.get_cmap(cmap) if isinstance(cmap, str) else cmap
    try:
        cmap_obj = cmap_obj.copy()
    except AttributeError:
        pass
    try:
        cmap_obj.set_bad("white")
    except AttributeError:
        pass
    return cmap_obj


def pcolor_r_z_on_axis(
    ax: plt.Axes,
    r: np.ndarray,
    z_over_r: np.ndarray,
    values: np.ndarray,
    cb_label: str,
    log_value: bool = False,
    overlay: Optional[Dict[str, np.ndarray]] = None,
    vmin: Optional[float] = None,
    vmax: Optional[float] = None,
    cmap: str = "magma",
    plot_entrap_surface: bool = True,
):
    """Plot a 2D r--z/r field on an existing axis using explicit cell edges."""
    R_edges, Z_edges = _rz_cell_edges(r, z_over_r)
    plot_values, log_prefix = _prepare_pcolor_values(values, log_value=log_value, vmin=vmin, vmax=vmax)
    cmap_obj = _copy_cmap_with_bad(cmap)
    norm = LogNorm(vmin=vmin, vmax=vmax) if log_value else Normalize(vmin=vmin, vmax=vmax)

    mesh = ax.pcolormesh(
        R_edges,
        Z_edges,
        plot_values,
        shading="flat",
        # vmin=vmin,
        # vmax=vmax,
        cmap=cmap_obj,
        rasterized=True,
        norm=norm,
    )
    ax.set_xscale("log")
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel(r"$z/r$")

    if overlay:
        ice_color_list = [
                        #   "white",
                        # [0.87058824, 0.84705882, 0.89411765],
                        [0.87058824, 0.85882353, 0.85490196], # CO
                        [0.79607843, 0.70588235, 0.95686275], # CO2
                        [0.36078431, 0.63137255, 0.87058824], # H2O
                        # [0.56470588, 0.33333333, 0.40784314], 
                        ]
        for i, (name, surf) in enumerate(overlay.items()):
            if not plot_entrap_surface and "pure" in name:
                name = name.split("pure ")[1]
            ax.plot(r, surf, label=name, color=ice_color_list[i], linewidth=1.4)
        ax.legend(fontsize=8, ncols=2, frameon=True)

    label = rf"{log_prefix}({cb_label})" if log_prefix else cb_label
    return mesh, label


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
    outpath = Path(outpath)

    fig, ax = plt.subplots(figsize=(9, 5))
    mesh, label = pcolor_r_z_on_axis(
        ax,
        r,
        z_over_r,
        values,
        cb_label=cb_label,
        log_value=log_value,
        overlay=overlay,
        vmin=vmin,
        vmax=vmax,
        cmap=cmap,
    )
    ax.set_title(title)
    cb = fig.colorbar(mesh, ax=ax)
    cb.set_label(label)
    fig.tight_layout()
    fig.savefig(outpath, dpi=220)
    plt.close(fig)


def make_2d_plots(data: Dict[str, np.ndarray], analysis_dir: Path, snap_index: int, 
                  plot_entrap_surface: bool = True) -> None:
    if not {"r_au", "z_over_r", "T_K"}.issubset(data):
        return

    r = data["r_au"]
    z_over_r = data["z_over_r"]

    overlay = {}
    for key in ["CO_pure", "CO_at_CO2", "CO_at_H2O", "CO2_pure", "CO2_at_H2O", "H2O"]:
        k = f"snow_surface_z_over_r_{key}"
        if not plot_entrap_surface and "_at_" in key:
            continue
        if k in data:
            overlay[key] = data[k]

    print("Plotting snow surfaces")
    pcolor_r_z(
        r, z_over_r, data["T_K"],
        f"Vertical temperature and release surfaces",# snapshot {snap_index}",
        "T [K]",
        f"{analysis_dir}/selected_2d_temperature_snow_surfaces.png",
        log_value=False,
        overlay=overlay,
        cmap=temp_cmap,
    )

    for key, title in [
        ("survival_CO_pure", r"CO pure survival"),
        ("survival_CO_at_CO2", r"CO@CO$_{2}$ survival"),
        ("survival_CO_at_H2O", r"CO@H$_{2}$O survival"),
        ("survival_CO2_pure", r"CO$_{2}$ pure survival"),
        ("survival_CO2_at_H2O", r"CO$_{2}$@H$_{2}$O survival"),
        ("survival_H2O", r"H$_{2}$O survival"),
    ]:
        if key in data:
            print(f"Plotting {key}")
            pcolor_r_z(
                r, z_over_r, data[key],
                f"{title}",# snapshot {snap_index}",
                r"Survival probability",
                analysis_dir / f"selected_2d_{key}.png",
                log_value=True,
                overlay=None,
                vmin=1e-4, vmax=1,
            )

    reservoir_maps = [
        ("surfbin_CO_gas", "CO gas"),
        ("surfbin_CO_pure_ice_pebble", "CO pure ice, pebbles"),
        ("surfbin_CO_at_CO2_ice_pebble", r"CO@CO$_2$ ice, pebbles"),
        ("surfbin_CO_at_H2O_ice_pebble", r"CO@H$_2$O ice, pebbles"),
        ("surfbin_CO_pure_ice_small", "CO pure ice, small grains"),
        ("surfbin_CO_at_CO2_ice_small", r"CO@CO$_{2}$ ice, small grains"),
        ("surfbin_CO_at_H2O_ice_small", r"CO@H$_{2}$O ice, small grains"),
        ("surfbin_CO2_gas", r"CO$_{2}$ gas"),
        ("surfbin_CO2_pure_ice_pebble", r"CO$_{2}$ pure ice, pebbles"),
        ("surfbin_CO2_at_H2O_ice_pebble", r"CO$_{2}$@H$_{2}$O ice, pebbles"),
        ("surfbin_CO2_pure_ice_small", r"CO$_{2}$ pure ice, small grains"),
        ("surfbin_CO2_at_H2O_ice_small", r"CO$_{2}$@H$_{2}$O ice, small grains"),
        ("surfbin_H2O_gas", r"H$_{2}$O gas"),
        ("surfbin_H2O_ice_pebble", r"H$_{2}$O ice, pebbles"),
        ("surfbin_H2O_ice_small", r"H$_{2}$O ice, small grains"),
    ]
    for key, title in reservoir_maps:
        if key in data:
            pcolor_r_z(
                r, z_over_r, data[key],
                f"{title}",# snapshot {snap_index}",
                r"vertical-bin surface density [g cm$^{-2}$]",
                Path(f"{analysis_dir}/selected_2d_{key}.png"),
                log_value=True,
                overlay=None,
                vmin=1e-12, vmax=1e-2,
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
               
    gas_frac = np.full_like(co_tot, np.nan, dtype=float)
    np.divide(
        data["surfbin_CO_gas"],
        co_tot,
        out=gas_frac,
        where=co_tot > co_floor,
    )
    gas_frac = np.clip(gas_frac, 0.0, 1.0)

    print("Plotting CO gas fraction")
    pcolor_r_z(
        r, z_over_r, gas_frac,
        f"CO gas fraction",# snapshot {snap_index}",
        r"$\Sigma_{\rm CO,gas}/\Sigma_{\rm CO}$",
        Path(f"{analysis_dir}/selected_2d_co_gas_frac.png"),
        log_value=True,
        overlay=None,
        vmin=1e-6,
        vmax=1e0,
    )
    print("Plotting matrix-associated CO fraction")
    pcolor_r_z(
        r, z_over_r, hidden_frac,
        f"Matrix-associated CO fraction",# snapshot {snap_index}",
        r"$\Sigma_{\rm CO,matrix}/\Sigma_{\rm CO}$",
        Path(f"{analysis_dir}/selected_2d_co_hidden_frac.png"),
        log_value=True,
        overlay=None,
        vmin=1e-6, vmax=1e0,
    )
    # pcolor_r_z(
    #     r,
    #     z_over_r,
    #     hidden_frac,
    #     f"Hidden CO fraction, snapshot {snap_index}",
    #     r"$\Sigma_{\rm CO,matrix}/\Sigma_{\rm CO}$",
    #     Path(f"{analysis_dir}/selected_2d_co_hidden_frac.png"),
    #     log_value=False,
    #     overlay=None,
    #     vmin=0.0,
    #     vmax=1.0,
    #     cmap="magma",
    # )


    bud = compute_volatile_budgets(data, collapse_vertical=False)
    
    co_tot_2d = bud["co_total"]
    small_volatile_2d = bud["small_volatile_ice"]
    pebble_volatile_2d = bud["pebble_volatile_ice"]
    small_c_o_2d = bud["C_over_O_small"]
    pebble_c_o_2d = bud["C_over_O_pebble"]

    z_over_r = data["z_over_r"]
    R = np.broadcast_to(r[:, None], z_over_r.shape)
    
    def plot_2d_quantity(quantity, title, cbar_label, savestr, log=False, vmin=None, vmax=None):
        fig, ax = plt.subplots(figsize=(8, 5))
        mesh, label = pcolor_r_z_on_axis(
            ax,
            r,
            z_over_r,
            quantity,
            cb_label=cbar_label,
            log_value=log,
            overlay=None,
            vmin=vmin,
            vmax=vmax,
            cmap="magma",
        )
        ax.set_title(title)
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(label)
        fig.tight_layout()
        fig.savefig(Path(savestr).with_suffix(".png"), dpi=220)
        plt.close(fig)

    print("Plotting small-grain volatile ice")
    plot_2d_quantity(
        small_volatile_2d,
        "Small-grain volatile ice",
        r"[g cm$^{-2}$ per vertical bin]",
        f"{analysis_dir}/small_ice",
        log=True,
    )
    
    print("Plotting pebble volatile ice")
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
    
    print("Plotting small-grain volatile C/O")
    plot_2d_quantity(
        small_c_o_2d_masked,
        "Small-grain volatile C/O",
        "C/O",
        f"{analysis_dir}/small_c_o",
        log=False,
        vmin=0,
        vmax=1,
    )
    
    print("Plotting pebble volatile C/O")
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
# Paper-facing summary plots
# -----------------------------
CO_CHANNEL_LABELS = {
    "CO_pure": "pure CO",
    "CO_at_CO2": r"CO@CO$_2$",
    "CO_at_H2O": r"CO@H$_2$O",
}

CO_CHANNEL_COLORS = {
    "CO_pure": color_list[0],
    "CO_at_CO2": color_list[1],
    "CO_at_H2O": color_list[2],
}

VOL_CHANNEL_LABELS = {
    "CO_pure": "pure CO",
    "CO_at_CO2": r"CO@CO$_2$",
    "CO_at_H2O": r"CO@H$_2$O",
    "CO2_pure": r"pure CO$_2$",
    "CO2_at_H2O": r"CO$_{2}$@H$_2$O",
    "H2O_pure": r"pure H$_2$O",
}

VOL_CHANNEL_COLORS = {
    "CO_pure": color_list[0],
    "CO_at_CO2": color_list[1],
    "CO_at_H2O": color_list[2],
    "CO2_pure": color_list[3],
    "CO2_at_H2O": color_list[4],
    "H2O_pure": color_list[5],
}


def make_paper_final_1d_summary(df: pd.DataFrame, analysis_dir: Path, snap_index: int) -> None:
    """
    Paper-facing radial summary for one selected snapshot:
      (a) CO partition by reservoir,
      (b) carrier-resolved volatile C/O.
    """
    if "r_au" not in df.columns:
        return

    df = add_volatile_budget_columns_1d(df)
    r = df["r_au"].to_numpy(dtype=float)

    co_gas = _col(df, "CO_gas")
    co_pure = _col(df, "CO_pure_ice_pebble") + _col(df, "CO_pure_ice_small")
    co_at_co2 = _col(df, "CO_at_CO2_ice_pebble") + _col(df, "CO_at_CO2_ice_small")
    co_at_h2o = _col(df, "CO_at_H2O_ice_pebble") + _col(df, "CO_at_H2O_ice_small")
    co_total = co_gas + co_pure + co_at_co2 + co_at_h2o

    denom = np.maximum(co_total, EPS)
    co_floor = 1.0e-10 * max(float(np.nanmax(co_total)), EPS)
    good = co_total > co_floor

    gas_frac = np.where(good, co_gas / denom, np.nan)
    pure_frac = np.where(good, co_pure / denom, np.nan)
    co2_frac = np.where(good, co_at_co2 / denom, np.nan)
    h2o_frac = np.where(good, co_at_h2o / denom, np.nan)
    hidden_frac = np.where(good, (co_at_co2 + co_at_h2o) / denom, np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8), constrained_layout=True)

    ax = axes[0]
    ax.semilogx(r, gas_frac, label="gas CO", color=color_list[5], linewidth=2.0)
    ax.semilogx(r, pure_frac, label="pure CO ice", color=CO_CHANNEL_COLORS["CO_pure"], linewidth=2.0)
    ax.semilogx(r, co2_frac, label=CO_CHANNEL_LABELS["CO_at_CO2"], color=CO_CHANNEL_COLORS["CO_at_CO2"], linewidth=2.0)
    ax.semilogx(r, h2o_frac, label=CO_CHANNEL_LABELS["CO_at_H2O"], color=CO_CHANNEL_COLORS["CO_at_H2O"], linewidth=2.0)
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel("Fraction of local CO")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(a) Local CO partitioning")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    ax = axes[1]
    ax.semilogx(r, df["C_over_O_pebble_masked"], label="pebble volatile C/O", color=color_list[0], linewidth=2.0)
    ax.semilogx(r, df["C_over_O_small_masked"], label="small-grain volatile C/O", color=color_list[1], linewidth=2.0)
    ax.semilogx(r, hidden_frac, label="matrix-associated CO fraction", color=color_list[2], linewidth=2.0, linestyle="--")
    ax.set_xlabel("Radius [au]")
    ax.set_ylabel("(C/O) or fraction")
    ax.set_ylim(-0.02, 1.02)
    ax.set_title("(b) Carrier composition of volatile ice")
    ax.legend(frameon=True, fontsize=8)
    ax.grid(True, which="both", alpha=0.25)

    fig.savefig(Path(analysis_dir) / "paper_radial_co_and_carrier_composition.png", dpi=240)
    plt.close(fig)


def _load_cumulative_release(
    snapshots: Sequence[SnapshotInfo],
    columns: Sequence[str],
) -> Tuple[Dict[str, np.ndarray], str, bool]:
    """Read cumulative release with version-aware compatibility handling."""
    if not snapshots:
        return {}, "missing", False

    final = read_snapshot(snapshots[-1].path)
    result: Dict[str, np.ndarray] = {}

    if any(f"cum_{column}" in final.columns for column in columns):
        for column in columns:
            cumulative_name = f"cum_{column}"
            if cumulative_name in final.columns:
                result[column] = final[cumulative_name].to_numpy(dtype=float)
        return result, "direct_cumulative_v2", True

    interval_v2 = "release_semantics_version" in final.columns or "release_interval_yr" in final.columns
    for column in columns:
        cumulative = None
        for snap in snapshots:
            df = read_snapshot(snap.path)
            if column not in df.columns:
                continue
            values = df[column].to_numpy(dtype=float)
            cumulative = values.copy() if cumulative is None else cumulative + values
        if cumulative is not None:
            result[column] = cumulative

    mode = "summed_intervals_v2" if interval_v2 else "legacy_snapshot_sample_v1"
    return result, mode, bool(interval_v2)


def make_paper_cumulative_release_profile(
    snapshots: Sequence[SnapshotInfo],
    analysis_dir: Path,
) -> None:
    """Paper-facing cumulative gross reservoir-loss profile."""
    if not snapshots:
        return

    df0 = read_snapshot(snapshots[0].path)
    if "r_au" not in df0.columns:
        return

    r = df0["r_au"].to_numpy(dtype=float)
    dlnr = _dlnr_from_centers(r)

    channels = [
        ("dM_CO_pure", "CO_pure"),
        ("dM_CO_at_CO2", "CO_at_CO2"),
        ("dM_CO_at_H2O", "CO_at_H2O"),
        ("dM_CO2_pure", "CO2_pure"),
        ("dM_CO2_at_H2O", "CO2_at_H2O"),
    ]
    cumulative, mode, reliable = _load_cumulative_release(
        snapshots, [column for column, _ in channels]
    )
    note = (
        f"release_mode={mode}\n"
        f"release_diagnostics_reliable={reliable}\n"
        "dM reservoir channels are gross positive losses during the phase update; "
        "they are not identical to the net gas source.\n"
    )
    (Path(analysis_dir) / "release_diagnostics_info.txt").write_text(note, encoding="utf-8")

    if not cumulative:
        return
    if not reliable:
        print(
            "WARNING: legacy release columns sample only saved phase steps; "
            "skipping the cumulative release figure. Rerun with the fixed model script."
        )
        return

    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    for column, channel in channels:
        if column not in cumulative:
            continue
        profile = cumulative[column] / np.maximum(dlnr, EPS) / MEARTH
        ax.semilogx(
            r,
            profile,
            label=VOL_CHANNEL_LABELS[channel],
            color=VOL_CHANNEL_COLORS[channel],
            linewidth=2.0,
        )

    ax.set_xlabel("Radius [au]")
    ax.set_ylabel(r"$dM_{\rm ice,gross}^{\rm cum}/d\ln r$ [$M_\oplus$]")
    ax.set_title("Cumulative gross loss from volatile-ice reservoirs")
    ax.legend(frameon=True)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(analysis_dir) / "paper_cumulative_release_profile.png", dpi=240)
    plt.close(fig)



def make_paper_2d_morphology(
    data: Dict[str, np.ndarray],
    analysis_dir: Path,
    snap_index: int,
    plot_entrap_surface: bool = False,
) -> None:
    """
    Paper-facing 2D morphology figure for the selected snapshot:
      (a) temperature and snow surfaces,
      (b) hidden CO fraction,
      (c) CO@CO2 surface density,
      (d) CO@H2O surface density.
    """
    required = {"r_au", "z_over_r", "T_K"}
    if not required.issubset(data):
        return

    r = data["r_au"]
    z_over_r = data["z_over_r"]

    snow_label_map = {
        "CO_pure": "pure CO",
        "CO_at_CO2": r"CO@CO$_2$",
        "CO_at_H2O": r"CO@H$_2$O",
        "CO2_pure": r"pure CO$_2$",
        "CO2_at_H2O": r"CO$_2$@H$_2$O",
        "H2O": r"H$_2$O",
    }
    overlay = {}
    for key, label in snow_label_map.items():
        k = f"snow_surface_z_over_r_{key}"
        if not plot_entrap_surface and "_at_" in key:
            continue
        if k in data:
            overlay[label] = data[k]

    co_at_co2 = _sum_fields(
        data,
        ["surfbin_CO_at_CO2_ice_pebble", "surfbin_CO_at_CO2_ice_small"],
    )
    co_at_h2o = _sum_fields(
        data,
        ["surfbin_CO_at_H2O_ice_pebble", "surfbin_CO_at_H2O_ice_small"],
    )
    co_hidden = co_at_co2 + co_at_h2o
    co_total = _sum_fields(
        data,
        [
            "surfbin_CO_gas",
            "surfbin_CO_pure_ice_pebble",
            "surfbin_CO_pure_ice_small",
            "surfbin_CO_at_CO2_ice_pebble",
            "surfbin_CO_at_CO2_ice_small",
            "surfbin_CO_at_H2O_ice_pebble",
            "surfbin_CO_at_H2O_ice_small",
        ],
    )

    co_floor = 1.0e-12 * max(float(np.nanmax(co_total)), EPS)
    hidden_frac = np.full_like(co_total, np.nan, dtype=float)
    np.divide(co_hidden, co_total, out=hidden_frac, where=co_total > co_floor)
    hidden_frac = np.clip(hidden_frac, 0.0, 1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12.0, 8.0), constrained_layout=True)

    panels = [
        (
            axes[0, 0],
            data["T_K"],
            "Temperature [K]",
            False,
            None,
            None,
            temp_cmap,
            overlay,
            "(a) Temperature and release surfaces",
        ),
        (
            axes[0, 1],
            hidden_frac,
            r"$\Sigma_{\rm CO,matrix}/\Sigma_{\rm CO}$",
            True,
            1e-6,
            1e0,
            "viridis",
            None,
            "(b) Matrix-associated CO fraction",
        ),
        (
            axes[1, 0],
            co_at_co2,
            r"$\Sigma_{\rm CO@CO_2}$ [g cm$^{-2}$]",
            True,
            1e-12,
            1e-2,
            "magma",
            None,
            r"(c) CO@CO$_2$ ice",
        ),
        (
            axes[1, 1],
            co_at_h2o,
            r"$\Sigma_{\rm CO@H_2O}$ [g cm$^{-2}$]",
            True,
            1e-12,
            1e-2,
            "magma",
            None,
            r"(d) CO@H$_2$O ice",
        ),
    ]

    for ax, values, cblabel, log_value, vmin, vmax, cmap, panel_overlay, title in panels:
        mesh, label = pcolor_r_z_on_axis(
            ax,
            r,
            z_over_r,
            values,
            cb_label=cblabel,
            log_value=log_value,
            overlay=panel_overlay,
            vmin=vmin,
            vmax=vmax,
            cmap=cmap,
            plot_entrap_surface=plot_entrap_surface and "snow surface" in title,
            
        )
        ax.set_title(title)
        cb = fig.colorbar(mesh, ax=ax)
        cb.set_label(label)

    fig.savefig(Path(analysis_dir) / "paper_fiducial_morphology.png", dpi=240)
    plt.close(fig)

def make_runtime_profile_plots(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Plot radial distributions of runtime-only cumulative diagnostics."""
    if "r_au" not in df.columns:
        return
    r = df["r_au"].to_numpy(dtype=float)
    dlnr = _dlnr_from_centers(r)
    cap_channels = [("CO_at_CO2", r"CO@CO$_2$"), ("CO_at_H2O", r"CO@H$_2$O"), ("CO2_at_H2O", r"CO$_2$@H$_2$O")]
    if any(f"cum_dM_capacity_excess_{ch}" in df.columns or f"current_dM_capacity_excess_{ch}" in df.columns for ch, _ in cap_channels):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), sharex=True)
        for i, (ch, label) in enumerate(cap_channels):
            current = df.get(f"current_dM_capacity_excess_{ch}", pd.Series(np.zeros_like(r))).to_numpy(dtype=float)
            gross = np.zeros_like(r)
            for prefix in ("initial", "cum"):
                c = f"{prefix}_dM_capacity_excess_{ch}"
                if c in df.columns: gross += df[c].to_numpy(dtype=float)
            axes[0].semilogx(r, current / np.maximum(dlnr, EPS) / MEARTH, label=label, color=color_list[i])
            axes[1].semilogx(r, gross / np.maximum(dlnr, EPS) / MEARTH, label=label, color=color_list[i])
        axes[0].set_title("Current capacity-limited target"); axes[0].set_ylabel(r"$dM_{\rm cap,reject}/d\ln r$ [$M_\oplus$]")
        axes[1].set_title("Cumulative gross capacity rejection"); axes[1].set_ylabel(r"$dM_{\rm cap,reject}^{\rm gross}/d\ln r$ [$M_\oplus$]")
        for ax in axes:
            ax.set_xlabel("Radius [au]"); ax.legend(fontsize=8); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "capacity_rejection_profile.png", dpi=220); plt.close(fig)

    needed = ["cum_dM_CO_gas_gain", "cum_dM_CO_gas_loss", "cum_dM_CO_gas"]
    if any(c in df.columns for c in needed):
        fig, ax = plt.subplots(figsize=(8.6, 4.8))
        for c, label, ls in (("cum_dM_CO_gas_gain", "positive gas gain", "-"), ("cum_dM_CO_gas_loss", "gas loss to solids", "--"), ("cum_dM_CO_gas", "net phase source", ":")):
            if c in df.columns: ax.semilogx(r, df[c].to_numpy(dtype=float) / np.maximum(dlnr, EPS) / MEARTH, label=label, linestyle=ls, linewidth=2)
        ax.axhline(0.0, color="0.5", linewidth=0.8); ax.set_xlabel("Radius [au]"); ax.set_ylabel(r"$dM_{\rm CO,phase}^{\rm cum}/d\ln r$ [$M_\oplus$]")
        ax.set_title("Cumulative CO phase exchange"); ax.legend(); ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(Path(analysis_dir) / "co_phase_exchange_profile.png", dpi=220); plt.close(fig)


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

    matrix_fraction_col = (
        "matrix_associated_CO_fraction"
        if "matrix_associated_CO_fraction" in df.columns
        else "hidden_CO_fraction"
    )
    if has_columns(df, ["r_au", matrix_fraction_col]):
        vals = df[matrix_fraction_col].to_numpy()
        j = int(np.nanargmax(vals))
        add("max_hidden_CO_fraction", float(np.nanmax(vals)), "Maximum radial matrix-associated CO fraction.")
        add("r_at_max_hidden_CO_fraction_au", float(df["r_au"].iloc[j]), "Radius of maximum matrix-associated CO fraction.")

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

    if diag is not None and not diag.empty:
        last = diag.iloc[-1]
        extra_cols = [
            "retained_CO_fraction", "retained_CO2_fraction", "retained_H2O_fraction",
            "cum_boundary_inner_CO_mearth", "cum_boundary_outer_CO_mearth",
            "current_capacity_excess_CO_at_CO2_mearth", "current_capacity_excess_CO_at_H2O_mearth", "current_capacity_excess_CO2_at_H2O_mearth",
            "current_capacity_active_cell_fraction_CO_at_CO2", "current_capacity_active_cell_fraction_CO_at_H2O", "current_capacity_active_cell_fraction_CO2_at_H2O",
            "total_capacity_excess_CO_at_CO2_mearth", "total_capacity_excess_CO_at_H2O_mearth", "total_capacity_excess_CO2_at_H2O_mearth",
            "current_capacity_to_gas_CO_mearth", "total_capacity_to_gas_CO_mearth", "phase_cycling_factor_CO",
            "cum_phase_gas_gain_CO_mearth", "cum_phase_gas_loss_CO_mearth", "cum_phase_gas_net_CO_mearth",
            "mass_balance_residual_fraction_CO", "mass_balance_residual_fraction_CO2", "mass_balance_residual_fraction_H2O",
            "cum_clipped_added_CO_mearth", "cum_clipped_added_CO2_mearth", "cum_clipped_added_H2O_mearth",
            "mass_weighted_epsilon_pebble", "mass_weighted_rel_delta_v_pebble_backreaction",
            "dt_min_yr", "dt_mean_yr", "dt_max_yr", "n_steps",
        ]
        for colname in extra_cols:
            if colname in diag.columns:
                add(f"final_{colname}", float(last[colname]), f"Final runtime diagnostic: {colname}.")

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
    parser.add_argument("--plot_entrap_surfaces", type=int, default=0, help="Plot entrapped snow surfaces.")
    parser.add_argument("--skip-2d", action="store_true", help="Skip all 2D NPZ analysis even when files exist.")
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

    ### for paper ####
    make_paper_final_1d_summary(selected_df, analysis_dir, selected.index)
    make_paper_cumulative_release_profile(snapshots, analysis_dir)

    ### additional diagnostics / appendix ####
    make_diagnostics_plots(diag, analysis_dir)
    make_final_1d_plots(selected_df, analysis_dir, selected.index, plot_entrap_surface=bool(args.plot_entrap_surfaces))
    make_runtime_profile_plots(selected_df, analysis_dir)

    if len(snapshots) > args.max_time_radius_snaps:
        indices = np.linspace(0, len(snapshots) - 1, args.max_time_radius_snaps).astype(int)
        tr_snaps = [snapshots[i] for i in np.unique(indices)]
    else:
        tr_snaps = snapshots
    make_time_radius_plots(tr_snaps, analysis_dir)

    snaps2d = [] if args.skip_2d else list_snapshots_2d(output_dir)
    if snaps2d:
        nearest = min(snaps2d, key=lambda s: abs(s.index - selected.index))
        data2d = read_2d_snapshot(nearest.path)

        ### for paper ####
        print("Plotting morphology...")
        make_paper_2d_morphology(data2d, analysis_dir, nearest.index, plot_entrap_surface=bool(args.plot_entrap_surfaces))
        make_paper_time_radius_co_partitioning(snapshots, analysis_dir)
        
        ### additional diagnostics / appendix ####
        print("Plotting 2D plots...")
        make_2d_plots(data2d, analysis_dir, nearest.index, plot_entrap_surface=bool(args.plot_entrap_surfaces))

    write_summary_metrics(selected_df, diag, analysis_dir / "summary_metrics.csv", selected.index)
    print(f"Analysis complete. Plots written to: {analysis_dir}")
    
    
if __name__ == "__main__":
    main()
