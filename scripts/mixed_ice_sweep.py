"""
mixed_ice_sweep.py

Generate and analyze a compact 18-run parameter sweep for the 1+1D mixed-ice
transport model.

This script does two things:

1. generate
   Takes an existing YAML skeleton as the default, modifies only selected
   parameters, and writes 18 sweep YAMLs with readable names.

2. analyze
   Reads the completed run outputs, extracts compact science metrics, and
   makes summary plots/tables designed to support the main paper-level claims.

Typical use
-----------

Generate YAMLs:

    python mixed_ice_sweep.py generate \
        --base-yaml example_mixed_ice_1p1d_params.yaml \
        --sweep-dir sweep_yamls \
        --run-root sweep_outputs \
        --model-script mixed_ice_transport_1p1d.py

Run the sweep:

    bash sweep_yamls/commands.sh

Analyze after runs complete:

    python mixed_ice_sweep.py analyze \
        --sweep-dir sweep_yamls \
        --analysis-dir sweep_analysis

Outputs from generate
---------------------

    sweep_yamls/
        fiducial.yaml
        ice_pure_snowline.yaml
        ...
        commands.txt
        commands.sh
        sweep_manifest.csv

Outputs from analyze
--------------------

    sweep_analysis/
        sweep_summary_metrics.csv
        sweep_summary_metrics.tex
        key_results.md
        ice_suite_release_r50.png
        ice_suite_co_partitioning.png
        stokes_sensitivity.png
        alpha_sensitivity.png
        condensation_sensitivity.png
        vertical_temperature_sensitivity.png
        vapor_diffusion_summary.png
        release_temperature_sensitivity.png
        cumulative_release_profiles_ice_suite.png

Assumptions
-----------

The model output directories contain:

    snapshots/snapshot_*.csv

Each snapshot has the usual 1+1D model columns, e.g.
CO_gas, CO_pure_ice_pebble, CO_at_CO2_ice_small, etc.

If release columns are present, e.g.
dM_CO_pure, dM_CO_at_CO2, dM_CO_at_H2O,
the script computes cumulative release radii. If not, release metrics are NaN.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import shlex
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml


AU = 1.495978707e13
MEARTH = 5.9722e27
EPS = 1.0e-300

MW = {
    "CO": 28.0101,
    "CO2": 44.0095,
    "H2O": 18.01528,
}

RELEASE_CHANNELS = ("CO_pure", "CO_at_CO2", "CO_at_H2O")


# ---------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------
def read_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def write_yaml(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)


def deep_update(base: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a nested dictionary without modifying base."""
    out = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def savefig(path: Path, dpi: int = 200) -> None:
    plt.tight_layout()
    plt.savefig(path, dpi=dpi)
    plt.close()


# ---------------------------------------------------------------------
# Sweep definition
# ---------------------------------------------------------------------
def make_run_specs() -> List[Dict[str, Any]]:
    """
    Compact 18-run sweep.

    Includes:
      - 5 ice-composition cases, including fiducial
      - 2 extra Stokes number cases
      - 2 extra alpha cases
      - 3 extra condensation-weight cases
      - 2 extra vertical-temperature cases
      - 2 vapor-diffusion-off cases
      - 2 release-temperature sensitivity cases
    """

    return [
        # ------------------------------------------------------------
        # Ice-composition suite
        # ------------------------------------------------------------
        {
            "name": "ice_pure_snowline",
            "group": "ice",
            "description": "Classical pure-snowline control.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 1.0,
                            "at_CO2": 0.0,
                            "at_H2O": 0.0,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 1.0,
                            "at_H2O": 0.0,
                        }
                    },
                }
            },
        },
        {
            "name": "ice_low_trap",
            "group": "ice",
            "description": "Mild trapping case.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.70,
                            "at_CO2": 0.25,
                            "at_H2O": 0.05,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.90,
                            "at_H2O": 0.10,
                        }
                    },
                }
            },
        },
        {
            "name": "fiducial",
            "group": "fiducial",
            "description": "Fiducial mixed-ice model.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.40,
                            "at_CO2": 0.40,
                            "at_H2O": 0.20,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.75,
                            "at_H2O": 0.25,
                        }
                    },
                }
            },
        },
        {
            "name": "ice_co2_trap",
            "group": "ice",
            "description": "CO mostly trapped in CO2-rich ice.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.20,
                            "at_CO2": 0.70,
                            "at_H2O": 0.10,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.75,
                            "at_H2O": 0.25,
                        }
                    },
                }
            },
        },
        {
            "name": "ice_h2o_trap",
            "group": "ice",
            "description": "Strong refractory trapping in H2O-rich ice.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.20,
                            "at_CO2": 0.20,
                            "at_H2O": 0.60,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.50,
                            "at_H2O": 0.50,
                        }
                    },
                }
            },
        },
        # ------------------------------------------------------------
        # Pebble Stokes number sensitivity
        # ------------------------------------------------------------
        {
            "name": "st_pebble_0p003",
            "group": "stokes",
            "description": "Slow/gas-coupled pebble drift.",
            "overrides": {
                "dust": {
                    "carriers": {
                        "pebble": {
                            "St": 0.003,
                        }
                    }
                }
            },
        },
        {
            "name": "st_pebble_0p1",
            "group": "stokes",
            "description": "Fast pebble drift.",
            "overrides": {
                "dust": {
                    "carriers": {
                        "pebble": {
                            "St": 0.1,
                        }
                    }
                }
            },
        },
        # ------------------------------------------------------------
        # Turbulence sensitivity
        # ------------------------------------------------------------
        {
            "name": "alpha_1e_m4",
            "group": "alpha",
            "description": "Low turbulence and low vertical mixing.",
            "overrides": {
                "gas": {
                    "alpha": 1.0e-4,
                },
                "vertical": {
                    "alpha_z": 1.0e-4,
                },
            },
        },
        {
            "name": "alpha_1e_m2",
            "group": "alpha",
            "description": "High turbulence and high vertical mixing.",
            "overrides": {
                "gas": {
                    "alpha": 1.0e-2,
                },
                "vertical": {
                    "alpha_z": 1.0e-2,
                },
            },
        },
        # ------------------------------------------------------------
        # Recondensation/carrier weighting sensitivity
        # ------------------------------------------------------------
        {
            "name": "cond_pebble_0p99",
            "group": "condensation",
            "description": "Recondensed vapor nearly all goes to pebbles.",
            "overrides": {
                "dust": {
                    "volatile_carrier_fractions": {
                        "pebble": 0.99,
                        "small": 0.01,
                    }
                }
            },
        },
        {
            "name": "cond_equal_0p50",
            "group": "condensation",
            "description": "Recondensed vapor split equally between carriers.",
            "overrides": {
                "dust": {
                    "volatile_carrier_fractions": {
                        "pebble": 0.50,
                        "small": 0.50,
                    }
                }
            },
        },
        {
            "name": "cond_small_0p90",
            "group": "condensation",
            "description": "Recondensed vapor mostly coats small grains.",
            "overrides": {
                "dust": {
                    "volatile_carrier_fractions": {
                        "pebble": 0.10,
                        "small": 0.90,
                    }
                }
            },
        },
        # ------------------------------------------------------------
        # Vertical temperature sensitivity
        # ------------------------------------------------------------
        {
            "name": "vertical_Tatm_1p2",
            "group": "vertical",
            "description": "Weakly warm atmosphere.",
            "overrides": {
                "vertical": {
                    "temperature": {
                        "T_atm_factor": 1.2,
                    }
                }
            },
        },
        {
            "name": "vertical_Tatm_3p0",
            "group": "vertical",
            "description": "Strongly warm atmosphere.",
            "overrides": {
                "vertical": {
                    "temperature": {
                        "T_atm_factor": 3.0,
                    }
                }
            },
        },
        # ------------------------------------------------------------
        # Vapor diffusion sensitivity
        # ------------------------------------------------------------
        {
            "name": "vdiff_fiducial_off",
            "group": "vapor_diffusion",
            "description": "Fiducial mixed ice with vapor diffusion disabled.",
            "overrides": {
                "volatiles": {
                    "include_vapor_diffusion": False,
                }
            },
        },
        {
            "name": "vdiff_h2o_trap_off",
            "group": "vapor_diffusion",
            "description": "H2O-trap case with vapor diffusion disabled.",
            "overrides": {
                "volatiles": {
                    "include_vapor_diffusion": False,
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.20,
                            "at_CO2": 0.20,
                            "at_H2O": 0.60,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.50,
                            "at_H2O": 0.50,
                        }
                    },
                }
            },
        },
        # ------------------------------------------------------------
        # Release-temperature sensitivity
        # ------------------------------------------------------------
        {
            "name": "release_cool",
            "group": "release_temperature",
            "description": "Cooler effective release temperatures.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "release_temperatures_K": {
                            "pure": 20.0,
                            "at_CO2": 60.0,
                            "at_H2O": 120.0,
                        }
                    },
                    "CO2": {
                        "release_temperatures_K": {
                            "pure": 60.0,
                            "at_H2O": 120.0,
                        }
                    },
                    "H2O": {
                        "release_temperature_K": 140.0,
                    },
                }
            },
        },
        {
            "name": "release_warm",
            "group": "release_temperature",
            "description": "Warmer effective release temperatures.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "release_temperatures_K": {
                            "pure": 30.0,
                            "at_CO2": 85.0,
                            "at_H2O": 150.0,
                        }
                    },
                    "CO2": {
                        "release_temperatures_K": {
                            "pure": 85.0,
                            "at_H2O": 150.0,
                        }
                    },
                    "H2O": {
                        "release_temperature_K": 170.0,
                    },
                }
            },
        },
        # ------------------------------------------------------------
        # Backreaction sensitivity
        # ------------------------------------------------------------
        {
            "name": "fiducial_w_backreact",
            "group": "backreact",
            "description": "Fiducial mixed-ice model with backreaction.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.40,
                            "at_CO2": 0.40,
                            "at_H2O": 0.20,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.75,
                            "at_H2O": 0.25,
                        }
                    },
                },
                "dust": {
                    "backreaction": {
                        "enabled": True,
                    }
                }
            },
        },
        {
            "name": "uncapped",
            "group": "uncapped",
            "description": "Fiducial mixed-ice model with uncapped trapping.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "solid_fractions": {
                            "pure": 0.40,
                            "at_CO2": 0.40,
                            "at_H2O": 0.20,
                        }
                    },
                    "CO2": {
                        "solid_fractions": {
                            "pure": 0.75,
                            "at_H2O": 0.25,
                        }
                    },
                    "trapping_capacity": {
                        "enabled": False
                    }
                },
            },
        },
    ]


def manifest_row(params: Dict[str, Any], name: str, group: str, description: str, yaml_path: Path) -> Dict[str, Any]:
    co = params["volatiles"]["CO"]["solid_fractions"]
    co2 = params["volatiles"]["CO2"]["solid_fractions"]
    cond = params["dust"]["volatile_carrier_fractions"]

    return {
        "run_name": name,
        "group": group,
        "description": description,
        "yaml_path": str(yaml_path),
        "output_dir": params["simulation"]["output_dir"],
        "CO_pure": co.get("pure", np.nan),
        "CO_at_CO2": co.get("at_CO2", np.nan),
        "CO_at_H2O": co.get("at_H2O", np.nan),
        "CO2_pure": co2.get("pure", np.nan),
        "CO2_at_H2O": co2.get("at_H2O", np.nan),
        "St_pebble": params["dust"]["carriers"]["pebble"]["St"],
        "St_small": params["dust"]["carriers"]["small"]["St"],
        "alpha": params["gas"]["alpha"],
        "alpha_z": params["vertical"]["alpha_z"],
        "cond_pebble": cond["pebble"],
        "cond_small": cond["small"],
        "T_atm_factor": params["vertical"]["temperature"]["T_atm_factor"],
        "Av_crit": params["vertical"]["shielding"]["Av_crit"],
        "vapor_diffusion": params["volatiles"]["include_vapor_diffusion"],
        "Trel_CO_pure": params["volatiles"]["CO"]["release_temperatures_K"]["pure"],
        "Trel_CO_at_CO2": params["volatiles"]["CO"]["release_temperatures_K"]["at_CO2"],
        "Trel_CO_at_H2O": params["volatiles"]["CO"]["release_temperatures_K"]["at_H2O"],
    }


def generate_yamls(
    base_yaml: Path,
    sweep_dir: Path,
    run_root: Path,
    model_script: str,
    save_2d: bool,
) -> None:
    base = read_yaml(base_yaml)
    ensure_dir(sweep_dir)
    ensure_dir(run_root)

    rows = []
    commands = []

    for spec in make_run_specs():
        name = spec["name"]
        params = deep_update(base, spec["overrides"])

        params.setdefault("simulation", {})
        params["simulation"]["name"] = name
        params["simulation"]["output_dir"] = str((run_root / name).resolve())
        params["simulation"]["overwrite"] = True

        params.setdefault("output", {})
        params["output"]["save_2d_npz"] = bool(save_2d)

        params["sweep_metadata"] = {
            "run_name": name,
            "group": spec["group"],
            "description": spec["description"],
        }

        yaml_path = sweep_dir / f"{name}.yaml"
        write_yaml(yaml_path, params)
        rows.append(manifest_row(params, name, spec["group"], spec["description"], yaml_path))

        commands.append(f"python {shlex.quote(model_script)} {shlex.quote(str(yaml_path))}")

    manifest_path = sweep_dir / "sweep_manifest.csv"
    pd.DataFrame(rows).to_csv(manifest_path, index=False)

    (sweep_dir / "commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    commands_sh = sweep_dir / "commands.sh"
    commands_sh.write_text("#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n".join(commands) + "\n", encoding="utf-8")
    os.chmod(commands_sh, 0o755)

    print(f"Wrote {len(rows)} YAMLs to {sweep_dir}")
    print(f"Wrote manifest to {manifest_path}")
    print(f"Wrote commands to {sweep_dir / 'commands.txt'} and {commands_sh}")


# ---------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------
def snapshot_paths(output_dir: Path) -> List[Path]:
    return sorted((output_dir / "snapshots").glob("snapshot_*.csv"))


def read_snapshot(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def log_edges_from_centers(r: np.ndarray) -> np.ndarray:
    r = np.asarray(r, dtype=float)
    lr = np.log(r)
    edges = np.empty(len(r) + 1)
    edges[1:-1] = 0.5 * (lr[:-1] + lr[1:])
    edges[0] = lr[0] - 0.5 * (lr[1] - lr[0])
    edges[-1] = lr[-1] + 0.5 * (lr[-1] - lr[-2])
    return np.exp(edges)


def annulus_area_cm2(r_au: np.ndarray) -> np.ndarray:
    e = log_edges_from_centers(r_au) * AU
    return math.pi * (e[1:] ** 2 - e[:-1] ** 2)


def dlnr_from_centers(r_au: np.ndarray) -> np.ndarray:
    e = log_edges_from_centers(r_au)
    return np.log(e[1:] / e[:-1])


def col(df: pd.DataFrame, name: str) -> np.ndarray:
    if name in df.columns:
        return df[name].to_numpy(dtype=float)
    return np.zeros(len(df), dtype=float)


def compute_budget(df: pd.DataFrame) -> Dict[str, np.ndarray]:
    co_peb = col(df, "CO_pure_ice_pebble") + col(df, "CO_at_CO2_ice_pebble") + col(df, "CO_at_H2O_ice_pebble")
    co_sml = col(df, "CO_pure_ice_small") + col(df, "CO_at_CO2_ice_small") + col(df, "CO_at_H2O_ice_small")
    co_gas = col(df, "CO_gas")
    co_hid = col(df, "CO_at_CO2_ice_pebble") + col(df, "CO_at_H2O_ice_pebble") + col(df, "CO_at_CO2_ice_small") + col(df, "CO_at_H2O_ice_small")

    co2_peb = col(df, "CO2_pure_ice_pebble") + col(df, "CO2_at_H2O_ice_pebble")
    co2_sml = col(df, "CO2_pure_ice_small") + col(df, "CO2_at_H2O_ice_small")
    co2_gas = col(df, "CO2_gas")

    h2o_peb = col(df, "H2O_ice_pebble")
    h2o_sml = col(df, "H2O_ice_small")
    h2o_gas = col(df, "H2O_gas")

    co_tot = co_peb + co_sml + co_gas
    peb_vol = co_peb + co2_peb + h2o_peb
    sml_vol = co_sml + co2_sml + h2o_sml
    sol_vol = peb_vol + sml_vol
    gas_vol = co_gas + co2_gas + h2o_gas
    tot_vol = sol_vol + gas_vol

    nco_peb = co_peb / MW["CO"]
    nco_sml = co_sml / MW["CO"]
    nco_gas = co_gas / MW["CO"]

    nco2_peb = co2_peb / MW["CO2"]
    nco2_sml = co2_sml / MW["CO2"]
    nco2_gas = co2_gas / MW["CO2"]

    nh2o_peb = h2o_peb / MW["H2O"]
    nh2o_sml = h2o_sml / MW["H2O"]
    nh2o_gas = h2o_gas / MW["H2O"]

    C_peb = nco_peb + nco2_peb
    O_peb = nco_peb + 2.0 * nco2_peb + nh2o_peb
    C_sml = nco_sml + nco2_sml
    O_sml = nco_sml + 2.0 * nco2_sml + nh2o_sml
    C_gas = nco_gas + nco2_gas
    O_gas = nco_gas + 2.0 * nco2_gas + nh2o_gas

    return {
        "co_pebble": co_peb,
        "co_small": co_sml,
        "co_gas": co_gas,
        "co_hidden": co_hid,
        "co_total": co_tot,
        "pebble_volatile": peb_vol,
        "small_volatile": sml_vol,
        "solid_volatile": sol_vol,
        "gas_volatile": gas_vol,
        "total_volatile": tot_vol,
        "C_pebble": C_peb,
        "O_pebble": O_peb,
        "C_small": C_sml,
        "O_small": O_sml,
        "C_gas": C_gas,
        "O_gas": O_gas,
        "C_over_O_pebble": C_peb / np.maximum(O_peb, EPS),
        "C_over_O_small": C_sml / np.maximum(O_sml, EPS),
        "C_over_O_gas": C_gas / np.maximum(O_gas, EPS),
        "hidden_CO_fraction": co_hid / np.maximum(co_tot, EPS),
        "gas_CO_fraction": co_gas / np.maximum(co_tot, EPS),
        "small_fraction_solid_volatile": sml_vol / np.maximum(sol_vol, EPS),
        "pebble_fraction_solid_volatile": peb_vol / np.maximum(sol_vol, EPS),
        "gas_fraction_total_volatile": gas_vol / np.maximum(tot_vol, EPS),
    }


def weighted_ratio(area: np.ndarray, numerator: np.ndarray, denominator: np.ndarray) -> float:
    return float(np.nansum(area * numerator) / max(np.nansum(area * denominator), EPS))


def weighted_mean(area: np.ndarray, values: np.ndarray, sigma_weights: np.ndarray) -> float:
    w = area * sigma_weights
    return float(np.nansum(w * values) / max(np.nansum(w), EPS))


def release_stats(r_au: np.ndarray, dM_cumulative: np.ndarray) -> Dict[str, float]:
    dM = np.nan_to_num(np.asarray(dM_cumulative, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    total = float(np.sum(dM))
    if total <= 0:
        return {"R_peak": np.nan, "R10": np.nan, "R50": np.nan, "R90": np.nan, "Mtot_Mearth": 0.0}

    dlnr = dlnr_from_centers(r_au)
    profile = dM / np.maximum(dlnr, EPS)
    R_peak = float(r_au[int(np.nanargmax(profile))])

    cum = np.cumsum(dM)
    R10 = float(np.interp(0.10 * total, cum, r_au))
    R50 = float(np.interp(0.50 * total, cum, r_au))
    R90 = float(np.interp(0.90 * total, cum, r_au))

    return {
        "R_peak": R_peak,
        "R10": R10,
        "R50": R50,
        "R90": R90,
        "Mtot_Mearth": total / MEARTH,
    }


def cumulative_release(snapshot_files: Sequence[Path]) -> Dict[str, Any]:
    if not snapshot_files:
        return {"have_release": False}

    df0 = read_snapshot(snapshot_files[0])
    r = df0["r_au"].to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "have_release": False,
        "r_au": r,
    }

    for channel in RELEASE_CHANNELS:
        out[f"dM_{channel}"] = np.zeros_like(r)

    out["dM_CO_gas"] = np.zeros_like(r)

    for snap in snapshot_files:
        df = read_snapshot(snap)
        for channel in RELEASE_CHANNELS:
            cname = f"dM_{channel}"
            if cname in df.columns:
                out[cname] += df[cname].to_numpy(dtype=float)
                out["have_release"] = True
        if "dM_CO_gas" in df.columns:
            out["dM_CO_gas"] += df["dM_CO_gas"].to_numpy(dtype=float)

    for channel in RELEASE_CHANNELS:
        stats = release_stats(r, out[f"dM_{channel}"])
        for key, value in stats.items():
            out[f"{key}_{channel}"] = value

    return out

def compute_classical_co_freezeout_proxy(
    df: pd.DataFrame,
    T_freeze_K: float = 20.0,
    transition_width_K: float = 2.0,
    temperature_column: str = "T_mid_K",
    ratio_floor_fraction: float = 1.0e-12,
) -> Dict[str, np.ndarray]:
    """
    Compare the model CO gas abundance to a classical pure-CO freeze-out proxy.

    This function uses the same total CO budget as the model, but repartitions it
    as if CO were only allowed to be either gas-phase CO or pure CO ice according
    to a simple temperature threshold.

    Classical prescription:
        f_gas_classical = 0.5 * [1 + tanh((T - T_freeze) / dT)]

        Sigma_CO_gas_classical = f_gas_classical * Sigma_CO_total
        Sigma_CO_ice_classical = (1 - f_gas_classical) * Sigma_CO_total

    The bias ratio is:
        B_CO = Sigma_CO_gas_model / Sigma_CO_gas_classical

    Interpretation:
        B_CO < 1  : model has less gas CO than classical freeze-out predicts
        B_CO ~ 1  : model agrees with classical freeze-out
        B_CO > 1  : model has more gas CO than classical freeze-out predicts

    Notes
    -----
    This is a 1D/midplane-temperature proxy. It is useful for sweep-level
    comparison, but it is not a full line-emitting-layer or RT diagnostic.
    """

    if temperature_column not in df.columns:
        raise KeyError(
            f"Could not find temperature column {temperature_column!r}. "
            f"Available columns include: {list(df.columns[:20])} ..."
        )

    T = df[temperature_column].to_numpy(dtype=float)

    # Total CO molecular budget in all reservoirs.
    CO_gas_model = col(df, "CO_gas")

    CO_pure_pebble = col(df, "CO_pure_ice_pebble")
    CO_pure_small = col(df, "CO_pure_ice_small")

    CO_at_CO2_pebble = col(df, "CO_at_CO2_ice_pebble")
    CO_at_CO2_small = col(df, "CO_at_CO2_ice_small")

    CO_at_H2O_pebble = col(df, "CO_at_H2O_ice_pebble")
    CO_at_H2O_small = col(df, "CO_at_H2O_ice_small")

    CO_solid_model = (
        CO_pure_pebble
        + CO_pure_small
        + CO_at_CO2_pebble
        + CO_at_CO2_small
        + CO_at_H2O_pebble
        + CO_at_H2O_small
    )

    CO_hidden_model = (
        CO_at_CO2_pebble
        + CO_at_CO2_small
        + CO_at_H2O_pebble
        + CO_at_H2O_small
    )

    CO_total = CO_gas_model + CO_solid_model

    # Smooth classical gas fraction.
    # Warm side -> gas. Cold side -> ice.
    dT = max(float(transition_width_K), 1.0e-12)
    f_gas_classical = 0.5 * (1.0 + np.tanh((T - float(T_freeze_K)) / dT))
    f_ice_classical = 1.0 - f_gas_classical

    CO_gas_classical = f_gas_classical * CO_total
    CO_ice_classical = f_ice_classical * CO_total

    # Local ratio. Use a floor to prevent meaningless blowups where the
    # classical expected gas column is essentially zero.
    denom_floor = ratio_floor_fraction * max(float(np.nanmax(CO_total)), 1.0e-300)
    B_CO = CO_gas_model / np.maximum(CO_gas_classical, denom_floor)

    delta_CO_gas = CO_gas_model - CO_gas_classical

    return {
        "T_K": T,
        "CO_total": CO_total,
        "CO_gas_model": CO_gas_model,
        "CO_solid_model": CO_solid_model,
        "CO_hidden_model": CO_hidden_model,
        "f_gas_classical": f_gas_classical,
        "f_ice_classical": f_ice_classical,
        "CO_gas_classical": CO_gas_classical,
        "CO_ice_classical": CO_ice_classical,
        "B_CO_gas_vs_classical": B_CO,
        "delta_CO_gas_vs_classical": delta_CO_gas,
        "hidden_CO_fraction": CO_hidden_model / np.maximum(CO_total, 1.0e-300),
        "model_gas_CO_fraction": CO_gas_model / np.maximum(CO_total, 1.0e-300),
        "classical_gas_CO_fraction": CO_gas_classical / np.maximum(CO_total, 1.0e-300),
    }
    
def summarize_classical_co_freezeout_proxy(
    df: pd.DataFrame,
    area: np.ndarray,
    T_freeze_K: float = 20.0,
    transition_width_K: float = 2.0,
) -> Dict[str, float]:
    """
    Area-integrated summary metrics for the classical CO freeze-out proxy.
    """

    fo = compute_classical_co_freezeout_proxy(
        df,
        T_freeze_K=T_freeze_K,
        transition_width_K=transition_width_K,
    )

    CO_gas_model_global = np.nansum(area * fo["CO_gas_model"])
    CO_gas_classical_global = np.nansum(area * fo["CO_gas_classical"])
    CO_total_global = np.nansum(area * fo["CO_total"])
    delta_global = np.nansum(area * fo["delta_CO_gas_vs_classical"])

    B_global = CO_gas_model_global / max(CO_gas_classical_global, 1.0e-300)

    # For local B statistics, only use places where classical gas CO is not tiny.
    mask = fo["CO_gas_classical"] > 1.0e-8 * max(float(np.nanmax(fo["CO_gas_classical"])), 1.0e-300)

    if np.any(mask):
        B_median = float(np.nanmedian(fo["B_CO_gas_vs_classical"][mask]))
        B_min = float(np.nanmin(fo["B_CO_gas_vs_classical"][mask]))
        B_max = float(np.nanmax(fo["B_CO_gas_vs_classical"][mask]))
    else:
        B_median = np.nan
        B_min = np.nan
        B_max = np.nan

    return {
        "B_CO_global_vs_classical_freezeout": float(B_global),
        "B_CO_median_where_classical_gas_exists": B_median,
        "B_CO_min_where_classical_gas_exists": B_min,
        "B_CO_max_where_classical_gas_exists": B_max,
        "CO_gas_model_global_Mearth": float(CO_gas_model_global / MEARTH),
        "CO_gas_classical_global_Mearth": float(CO_gas_classical_global / MEARTH),
        "delta_CO_gas_vs_classical_global_Mearth": float(delta_global / MEARTH),
        "model_global_gas_CO_fraction": float(CO_gas_model_global / max(CO_total_global, 1.0e-300)),
        "classical_global_gas_CO_fraction": float(CO_gas_classical_global / max(CO_total_global, 1.0e-300)),
    }

def analyze_run(row: pd.Series) -> Dict[str, Any]:
    result = row.to_dict()
    result["completed"] = False
    result["status"] = "missing"

    outdir = Path(row["output_dir"])
    snaps = snapshot_paths(outdir)
    if not snaps:
        result["status"] = "missing_snapshots"
        return result

    final_df = read_snapshot(snaps[-1])
    r = final_df["r_au"].to_numpy(dtype=float)
    area = annulus_area_cm2(r)
    bud = compute_budget(final_df)
    
    freezeout = summarize_classical_co_freezeout_proxy(
        final_df,
        area,
        T_freeze_K=20.0,
        transition_width_K=2.0,
    )

    result.update(freezeout)

    result["completed"] = True
    result["status"] = "ok"
    result["n_snapshots"] = len(snaps)
    result["final_time_yr"] = float(final_df["time_yr"].iloc[0]) if "time_yr" in final_df.columns else np.nan

    co_mass = np.sum(area * bud["co_total"])
    hidden_mass = np.sum(area * bud["co_hidden"])
    gas_co_mass = np.sum(area * bud["co_gas"])
    solid_volatile_mass = np.sum(area * bud["solid_volatile"])
    small_volatile_mass = np.sum(area * bud["small_volatile"])
    pebble_volatile_mass = np.sum(area * bud["pebble_volatile"])
    total_volatile_mass = np.sum(area * bud["total_volatile"])
    gas_volatile_mass = np.sum(area * bud["gas_volatile"])

    result["final_hidden_CO_fraction"] = float(hidden_mass / max(co_mass, EPS))
    result["final_gas_CO_fraction"] = float(gas_co_mass / max(co_mass, EPS))
    result["max_hidden_CO_fraction"] = float(np.nanmax(bud["hidden_CO_fraction"]))
    result["max_gas_CO_fraction"] = float(np.nanmax(bud["gas_CO_fraction"]))

    result["global_small_fraction_solid_volatile"] = float(small_volatile_mass / max(solid_volatile_mass, EPS))
    result["global_pebble_fraction_solid_volatile"] = float(pebble_volatile_mass / max(solid_volatile_mass, EPS))
    result["global_gas_fraction_total_volatile"] = float(gas_volatile_mass / max(total_volatile_mass, EPS))
    result["max_small_fraction_solid_volatile"] = float(np.nanmax(bud["small_fraction_solid_volatile"]))

    result["global_C_over_O_pebble"] = weighted_ratio(area, bud["C_pebble"], bud["O_pebble"])
    result["global_C_over_O_small"] = weighted_ratio(area, bud["C_small"], bud["O_small"])
    result["global_C_over_O_gas"] = weighted_ratio(area, bud["C_gas"], bud["O_gas"])
    result["delta_C_over_O_small_minus_pebble"] = result["global_C_over_O_small"] - result["global_C_over_O_pebble"]

    result["mass_weighted_radial_C_over_O_pebble"] = weighted_mean(area, bud["C_over_O_pebble"], bud["pebble_volatile"])
    result["mass_weighted_radial_C_over_O_small"] = weighted_mean(area, bud["C_over_O_small"], bud["small_volatile"])

    if "Sigma_g" in final_df.columns:
        Sigma_g = final_df["Sigma_g"].to_numpy(dtype=float)
        result["max_hidden_CO_over_Sigma_g"] = float(np.nanmax(bud["co_hidden"] / np.maximum(Sigma_g, EPS)))
        result["max_solid_volatile_over_Sigma_g"] = float(np.nanmax(bud["solid_volatile"] / np.maximum(Sigma_g, EPS)))
    else:
        result["max_hidden_CO_over_Sigma_g"] = np.nan
        result["max_solid_volatile_over_Sigma_g"] = np.nan

    rel = cumulative_release(snaps)
    result["have_release"] = bool(rel["have_release"])
    for channel in RELEASE_CHANNELS:
        for metric in ("R_peak", "R10", "R50", "R90", "Mtot_Mearth"):
            result[f"{metric}_{channel}"] = rel.get(f"{metric}_{channel}", np.nan)

    result["total_CO_release_Mearth"] = float(
        sum(result.get(f"Mtot_Mearth_{ch}", 0.0) for ch in RELEASE_CHANNELS)
    )

    return result


# ---------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------
def ordered(df: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    order = {name: i for i, name in enumerate(names)}
    return (
        df[df["run_name"].isin(names)]
        .assign(_order=lambda x: x["run_name"].map(order))
        .sort_values("_order")
        .drop(columns="_order")
    )


def include_fiducial(df: pd.DataFrame, group: str) -> pd.DataFrame:
    return df[(df["group"] == group) | (df["run_name"] == "fiducial")].copy()


def plot_ice_release(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["ice_pure_snowline", "ice_low_trap", "fiducial", "ice_co2_trap", "ice_h2o_trap", "fiducial_w_backreact", "uncapped"]
    sub = ordered(df, names)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.25

    plt.figure(figsize=(10, 5))
    plt.bar(x - w, sub["R50_CO_pure"], width=w, label="pure CO")
    plt.bar(x, sub["R50_CO_at_CO2"], width=w, label="CO@CO2")
    plt.bar(x + w, sub["R50_CO_at_H2O"], width=w, label="CO@H2O")
    plt.yscale("log")
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel(r"Median release radius $R_{50}$ [au]")
    plt.title("Ice matrix controls where CO enters the gas")
    plt.legend()
    plt.grid(True, which="both", axis="y", alpha=0.3)
    savefig(analysis_dir / "ice_suite_release_r50.png")


def plot_ice_partitioning(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["ice_pure_snowline", "ice_low_trap", "fiducial", "ice_co2_trap", "ice_h2o_trap", "fiducial_w_backreact", "uncapped"]
    sub = ordered(df, names)
    print(sub)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, sub["final_hidden_CO_fraction"], width=w, label="hidden CO")
    plt.bar(x + w / 2, sub["final_gas_CO_fraction"], width=w, label="gas CO")
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel("Fraction of total CO")
    plt.ylim(0, 1)
    plt.title("Final CO partitioning")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    savefig(analysis_dir / "ice_suite_co_partitioning.png")


def plot_cumulative_release_profiles(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["ice_pure_snowline", "ice_low_trap", "fiducial", "ice_co2_trap", "ice_h2o_trap", "fiducial_w_backreact", "uncapped"]
    sub = ordered(df, names)
    if sub.empty:
        return

    for _, row in sub.iterrows():
        outdir = Path(row["output_dir"])
        snaps = snapshot_paths(outdir)
        rel = cumulative_release(snaps)
        if not rel.get("have_release", False):
            continue

        r = rel["r_au"]
        dlnr = dlnr_from_centers(r)
        plt.figure(figsize=(9, 5))
        for channel, label in [
            ("CO_pure", "pure CO"),
            ("CO_at_CO2", "CO@CO2"),
            ("CO_at_H2O", "CO@H2O"),
        ]:
            prof = rel[f"dM_{channel}"] / np.maximum(dlnr, EPS) / MEARTH
            plt.semilogx(r, prof, label=label)
        plt.xlabel("Radius [au]")
        plt.ylabel(r"$dM_{\rm CO,release}^{\rm cum}/d\ln r$ [M$_\oplus$]")
        plt.title(f"Cumulative CO release: {row['run_name']}")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        savefig(analysis_dir / f"cumulative_release_{row['run_name']}.png")


def plot_transport(df: pd.DataFrame, analysis_dir: Path) -> None:
    sub = include_fiducial(df, "stokes").sort_values("St_pebble")
    if not sub.empty:
        plt.figure(figsize=(8, 5))
        for colname, label in [
            ("R50_CO_pure", "pure CO"),
            ("R50_CO_at_CO2", "CO@CO2"),
            ("R50_CO_at_H2O", "CO@H2O"),
        ]:
            plt.plot(sub["St_pebble"], sub[colname], marker="o", label=label)
        plt.xscale("log")
        plt.yscale("log")
        plt.xlabel("Pebble Stokes number")
        plt.ylabel(r"Median release radius $R_{50}$ [au]")
        plt.title("Sensitivity to pebble drift")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        savefig(analysis_dir / "stokes_sensitivity.png")

    sub = include_fiducial(df, "alpha").sort_values("alpha")
    if not sub.empty:
        plt.figure(figsize=(8, 5))
        plt.plot(sub["alpha"], sub["final_hidden_CO_fraction"], marker="o", label="hidden CO fraction")
        plt.plot(sub["alpha"], sub["final_gas_CO_fraction"], marker="o", label="gas CO fraction")
        plt.plot(sub["alpha"], sub["global_small_fraction_solid_volatile"], marker="o", label="small volatile fraction")
        plt.xscale("log")
        plt.xlabel(r"$\alpha$ and $\alpha_z$")
        plt.ylabel("Fraction")
        plt.title("Sensitivity to turbulence")
        plt.legend()
        plt.grid(True, which="both", alpha=0.3)
        savefig(analysis_dir / "alpha_sensitivity.png")


def plot_condensation(df: pd.DataFrame, analysis_dir: Path) -> None:
    sub = include_fiducial(df, "condensation").sort_values("cond_small")
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(sub["cond_small"], sub["global_C_over_O_pebble"], marker="o", label="pebble C/O")
    plt.plot(sub["cond_small"], sub["global_C_over_O_small"], marker="o", label="small-grain C/O")
    plt.plot(sub["cond_small"], sub["global_small_fraction_solid_volatile"], marker="o", label="small volatile fraction")
    plt.xlabel("Small-grain condensation weight")
    plt.ylabel("Ratio / fraction")
    plt.title("Carrier history and recondensation sensitivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(analysis_dir / "condensation_sensitivity.png")
    
    
def plot_freezeout(df: pd.DataFrame, analysis_dir: Path) -> None:
    """
    Plot sweep-level diagnostics comparing model CO gas to a classical
    pure-CO freeze-out prescription.

    Requires analyze_run(...) to have added columns from:
        summarize_classical_co_freezeout_proxy(...)
    """

    required = [
        "B_CO_global_vs_classical_freezeout",
        "model_global_gas_CO_fraction",
        "classical_global_gas_CO_fraction",
    ]

    if not all(c in df.columns for c in required):
        print(
            "Skipping freeze-out plots because required columns are missing. "
            "Did you add result.update(summarize_classical_co_freezeout_proxy(...)) "
            "inside analyze_run(...) ?"
        )
        return

    # ------------------------------------------------------------
    # Plot 1: global B_CO ratio for all completed runs
    # ------------------------------------------------------------
    sub = df.copy()
    sub = sub.sort_values("B_CO_global_vs_classical_freezeout")

    plt.figure(figsize=(10, 6))
    plt.barh(
        sub["run_name"],
        sub["B_CO_global_vs_classical_freezeout"],
    )
    plt.axvline(1.0, linewidth=1)
    plt.xscale("log")
    plt.xlabel(
        r"$B_{\rm CO} = M_{\rm CO,gas}^{\rm model} / "
        r"M_{\rm CO,gas}^{\rm classical}$"
    )
    plt.ylabel("Run")
    plt.title("Global CO gas bias relative to classical freeze-out")
    plt.grid(True, which="both", axis="x", alpha=0.3)
    savefig(analysis_dir / "freezeout_B_CO_global.png")

    # ------------------------------------------------------------
    # Plot 2: model gas fraction versus classical gas fraction
    # ------------------------------------------------------------
    plt.figure(figsize=(8, 5))
    plt.scatter(
        sub["classical_global_gas_CO_fraction"],
        sub["model_global_gas_CO_fraction"],
    )

    lo = min(
        np.nanmin(sub["classical_global_gas_CO_fraction"]),
        np.nanmin(sub["model_global_gas_CO_fraction"]),
    )
    hi = max(
        np.nanmax(sub["classical_global_gas_CO_fraction"]),
        np.nanmax(sub["model_global_gas_CO_fraction"]),
    )

    lo = max(lo, 1e-6)
    hi = max(hi, lo * 10)

    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1)

    for _, row in sub.iterrows():
        plt.annotate(
            row["run_name"],
            (
                row["classical_global_gas_CO_fraction"],
                row["model_global_gas_CO_fraction"],
            ),
            fontsize=7,
            alpha=0.8,
        )

    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Classical freeze-out gas CO fraction")
    plt.ylabel("Model gas CO fraction")
    plt.title("Model versus classical gas-phase CO fraction")
    plt.grid(True, which="both", alpha=0.3)
    savefig(analysis_dir / "freezeout_model_vs_classical_gas_fraction.png")

    # ------------------------------------------------------------
    # Plot 3: ice-composition subset, easier to interpret
    # ------------------------------------------------------------
    ice_names = [
        "ice_pure_snowline",
        "ice_low_trap",
        "fiducial",
        "ice_co2_trap",
        "ice_h2o_trap",
    ]

    ice = ordered(sub, ice_names)

    if not ice.empty:
        x = np.arange(len(ice))
        w = 0.35

        plt.figure(figsize=(10, 5))
        plt.bar(
            x - w / 2,
            ice["model_global_gas_CO_fraction"],
            width=w,
            label="model",
        )
        plt.bar(
            x + w / 2,
            ice["classical_global_gas_CO_fraction"],
            width=w,
            label="classical freeze-out proxy",
        )
        plt.xticks(x, ice["run_name"], rotation=30, ha="right")
        plt.ylabel("Global gas CO fraction")
        plt.title("Gas CO fraction: model versus classical freeze-out")
        plt.legend()
        plt.grid(True, axis="y", alpha=0.3)
        savefig(analysis_dir / "freezeout_ice_suite_gas_fraction.png")

    # ------------------------------------------------------------
    # Plot 4: delta gas CO mass, if available
    # ------------------------------------------------------------
    if "delta_CO_gas_vs_classical_global_Mearth" in sub.columns:
        sub2 = sub.sort_values("delta_CO_gas_vs_classical_global_Mearth")

        plt.figure(figsize=(10, 6))
        plt.barh(
            sub2["run_name"],
            sub2["delta_CO_gas_vs_classical_global_Mearth"],
        )
        plt.axvline(0.0, linewidth=1)
        plt.xlabel(
            r"$M_{\rm CO,gas}^{\rm model} - "
            r"M_{\rm CO,gas}^{\rm classical}$ [M$_\oplus$]"
        )
        plt.ylabel("Run")
        plt.title("Global gas CO excess/depletion relative to classical freeze-out")
        plt.grid(True, axis="x", alpha=0.3)
        savefig(analysis_dir / "freezeout_delta_CO_gas_global.png")

def plot_vertical(df: pd.DataFrame, analysis_dir: Path) -> None:
    sub = include_fiducial(df, "vertical").sort_values("T_atm_factor")
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(sub["T_atm_factor"], sub["final_hidden_CO_fraction"], marker="o", label="hidden CO fraction")
    plt.plot(sub["T_atm_factor"], sub["global_C_over_O_pebble"], marker="o", label="pebble C/O")
    plt.plot(sub["T_atm_factor"], sub["global_C_over_O_small"], marker="o", label="small-grain C/O")
    plt.xlabel(r"$T_{\rm atm}/T_{\rm mid}$")
    plt.ylabel("Ratio / fraction")
    plt.title("Vertical temperature sensitivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(analysis_dir / "vertical_temperature_sensitivity.png")


def plot_vdiff(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["fiducial", "vdiff_fiducial_off", "ice_h2o_trap", "vdiff_h2o_trap_off"]
    sub = ordered(df, names)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, sub["final_hidden_CO_fraction"], width=w, label="hidden CO")
    plt.bar(x + w / 2, sub["final_gas_CO_fraction"], width=w, label="gas CO")
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel("Fraction of total CO")
    plt.ylim(0, 1)
    plt.title("Vapor diffusion sensitivity")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    savefig(analysis_dir / "vapor_diffusion_summary.png")


def plot_release_temperature(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["release_cool", "fiducial", "release_warm"]
    sub = ordered(df, names)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar(x - w, sub["R50_CO_pure"], width=w, label="pure CO")
    plt.bar(x, sub["R50_CO_at_CO2"], width=w, label="CO@CO2")
    plt.bar(x + w, sub["R50_CO_at_H2O"], width=w, label="CO@H2O")
    plt.yscale("log")
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel(r"Median release radius $R_{50}$ [au]")
    plt.title("Effective release-temperature sensitivity")
    plt.legend()
    plt.grid(True, which="both", axis="y", alpha=0.3)
    savefig(analysis_dir / "release_temperature_sensitivity.png")


def write_key_results(metrics: pd.DataFrame, analysis_dir: Path) -> None:
    lines = ["# Key sweep results", ""]
    ok = metrics[metrics["completed"] == True].copy()
    if ok.empty:
        lines.append("No completed runs found.")
        (analysis_dir / "key_results.md").write_text("\n".join(lines), encoding="utf-8")
        return

    fid = ok[ok["run_name"] == "fiducial"]
    if not fid.empty:
        row = fid.iloc[0]
        lines += [
            "## Fiducial",
            f"- Final hidden CO fraction: {row['final_hidden_CO_fraction']:.3f}",
            f"- Final gas CO fraction: {row['final_gas_CO_fraction']:.3f}",
            f"- Global pebble C/O: {row['global_C_over_O_pebble']:.3f}",
            f"- Global small-grain C/O: {row['global_C_over_O_small']:.3f}",
            f"- Small-grain fraction of solid volatile ice: {row['global_small_fraction_solid_volatile']:.3f}",
            f"- R50 release radii [au]: pure CO={row['R50_CO_pure']:.3g}, CO@CO2={row['R50_CO_at_CO2']:.3g}, CO@H2O={row['R50_CO_at_H2O']:.3g}",
            "",
        ]

    if "final_hidden_CO_fraction" in ok.columns:
        row = ok.loc[ok["final_hidden_CO_fraction"].idxmax()]
        lines.append(f"- Largest hidden CO fraction: `{row['run_name']}` = {row['final_hidden_CO_fraction']:.3f}")

    if "delta_C_over_O_small_minus_pebble" in ok.columns:
        row = ok.loc[ok["delta_C_over_O_small_minus_pebble"].idxmax()]
        lines.append(f"- Largest small-minus-pebble C/O offset: `{row['run_name']}` = {row['delta_C_over_O_small_minus_pebble']:.3f}")

    if "total_CO_release_Mearth" in ok.columns:
        row = ok.loc[ok["total_CO_release_Mearth"].idxmax()]
        lines.append(f"- Largest cumulative CO release: `{row['run_name']}` = {row['total_CO_release_Mearth']:.3e} M_Earth")

    lines += [
        "",
        "## What to look for",
        "- The ice-composition suite should show whether CO release moves inward as more CO is tied to CO2/H2O matrices.",
        "- The Stokes-number sensitivity tests whether drift is needed to deliver trapped CO inward.",
        "- The condensation-weight suite tests whether small-grain C/O is robust or dominated by freeze-out surface-area assumptions.",
        "- The vapor-diffusion runs test whether CO gas redistribution is local or diffusion-controlled.",
    ]

    (analysis_dir / "key_results.md").write_text("\n".join(lines), encoding="utf-8")


def analyze_sweep(sweep_dir: Path, analysis_dir: Path) -> None:
    manifest_path = sweep_dir / "sweep_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    ensure_dir(analysis_dir)
    manifest = pd.read_csv(manifest_path)

    rows = []
    for _, row in manifest.iterrows():
        rows.append(analyze_run(row))

    metrics = pd.DataFrame(rows)
    metrics_path = analysis_dir / "sweep_summary_metrics.csv"
    metrics.to_csv(metrics_path, index=False)

    # Small LaTeX table of key metrics.
    table_cols = [
        "run_name",
        "group",
        "final_hidden_CO_fraction",
        "final_gas_CO_fraction",
        "global_C_over_O_pebble",
        "global_C_over_O_small",
        "R50_CO_pure",
        "R50_CO_at_CO2",
        "R50_CO_at_H2O",
        "B_CO_global_vs_classical_freezeout",
        "delta_CO_gas_vs_classical_global_Mearth",
        "model_global_gas_CO_fraction",
        "classical_global_gas_CO_fraction",
    ]
    available_cols = [c for c in table_cols if c in metrics.columns]
    with open(analysis_dir / "sweep_summary_metrics.tex", "w", encoding="utf-8") as f:
        f.write(metrics[available_cols].to_latex(index=False, float_format="%.3g"))

    ok = metrics[metrics["completed"] == True].copy()
    if ok.empty:
        print(f"No completed runs found. Wrote metrics table to {metrics_path}")
        return

    plot_ice_release(ok, analysis_dir)
    plot_ice_partitioning(ok, analysis_dir)
    plot_cumulative_release_profiles(ok, analysis_dir)
    plot_transport(ok, analysis_dir)
    plot_condensation(ok, analysis_dir)
    plot_vertical(ok, analysis_dir)
    plot_vdiff(ok, analysis_dir)
    plot_release_temperature(ok, analysis_dir)
    plot_freezeout(ok, analysis_dir)
    write_key_results(ok, analysis_dir)

    print(f"Analyzed {len(ok)} completed runs out of {len(metrics)} total.")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote plots/tables to: {analysis_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/analyze an 18-run mixed-ice sweep.")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("generate", help="Generate sweep YAMLs and command files.")
    p.add_argument("--base-yaml", required=True, type=str, help="Existing YAML skeleton to use as default.")
    p.add_argument("--sweep-dir", default="sweep_yamls", type=str, help="Output directory for generated YAMLs.")
    p.add_argument("--run-root", default="sweep_outputs", type=str, help="Root directory for model outputs.")
    p.add_argument("--model-script", default="mixed_ice_transport_1p1d.py", type=str, help="Model script to use in commands file.")
    p.add_argument("--save-2d", action="store_true", help="Keep 2D npz outputs enabled for all sweep runs. Default disables 2D output.")

    p = sub.add_parser("analyze", help="Analyze completed sweep outputs.")
    p.add_argument("--sweep-dir", default="sweep_yamls", type=str, help="Directory containing sweep_manifest.csv.")
    p.add_argument("--analysis-dir", default="sweep_analysis", type=str, help="Output directory for plots and tables.")

    p = sub.add_parser("all", help="Generate YAMLs, then analyze any already completed outputs.")
    p.add_argument("--base-yaml", required=True, type=str, help="Existing YAML skeleton to use as default.")
    p.add_argument("--sweep-dir", default="sweep_yamls", type=str, help="Output directory for generated YAMLs.")
    p.add_argument("--run-root", default="sweep_outputs", type=str, help="Root directory for model outputs.")
    p.add_argument("--model-script", default="mixed_ice_transport_1p1d.py", type=str, help="Model script to use in commands file.")
    p.add_argument("--analysis-dir", default="sweep_analysis", type=str, help="Output directory for plots and tables.")
    p.add_argument("--save-2d", action="store_true", help="Keep 2D npz outputs enabled for all sweep runs. Default disables 2D output.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.command == "generate":
        generate_yamls(
            base_yaml=Path(args.base_yaml).expanduser().resolve(),
            sweep_dir=Path(args.sweep_dir).expanduser().resolve(),
            run_root=Path(args.run_root).expanduser().resolve(),
            model_script=args.model_script,
            save_2d=args.save_2d,
        )

    elif args.command == "analyze":
        analyze_sweep(
            sweep_dir=Path(args.sweep_dir).expanduser().resolve(),
            analysis_dir=Path(args.analysis_dir).expanduser().resolve(),
        )

    elif args.command == "all":
        generate_yamls(
            base_yaml=Path(args.base_yaml).expanduser().resolve(),
            sweep_dir=Path(args.sweep_dir).expanduser().resolve(),
            run_root=Path(args.run_root).expanduser().resolve(),
            model_script=args.model_script,
            save_2d=args.save_2d,
        )
        analyze_sweep(
            sweep_dir=Path(args.sweep_dir).expanduser().resolve(),
            analysis_dir=Path(args.analysis_dir).expanduser().resolve(),
        )


if __name__ == "__main__":
    main()
