"""
mixed_ice_sweep.py

Generate and analyze a compact 21-run parameter sweep for the 1+1D mixed-ice
transport model.

This script does two things:

1. generate
   Takes an existing YAML skeleton as the default, modifies only selected
   parameters, and writes 21 sweep YAMLs with readable names.

2. analyze
   Reads the completed run outputs, extracts compact science metrics, and
   makes summary plots/tables designed to support the main paper-level claims.

Typical use
-----------

Generate YAMLs:

    python mixed_ice_sweep.py generate \
        --base-yaml example_mist_params.yaml \
        --sweep-dir sweep_yamls \
        --run-root sweep_outputs \
        --model-script mixed_ice_transport_1p1d.py\
        --save-2d

Run the sweep:

    bash sweep_yamls/commands.sh

Analyze after runs complete:

    python mixed_ice_sweep.py analyze \
        --sweep-dir sweep_yamls \
        --analysis-dir sweep_analysis
        
Single command, e.g.:
python mixed_ice_sweep.py generate --base-yaml example_mist_params.yaml --save-2d --sweep-dir sweep_yamls --run-root sweep_outputs --model-script mixed_ice_transport_1p1d.py
bash sweep_yamls/commands.sh
python mixed_ice_sweep.py analyze --sweep-dir sweep_yamls --analysis-dir sweep_analysis

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
        paper_ice_suite_summary.png
        paper_cumulative_release_fiducial.png
        paper_sensitivity_summary.png
        ice_suite_co_budget_stacked.png
        cumulative_release_profiles_ice_suite.png
        runtime_process_summary_ice_suite.png
        current_capacity_saturation_ice_suite.png
        sweep_numerical_quality.png

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
try:
    from colorspacious import cspace_converter
except ImportError:  # Optional plotting enhancement.
    cspace_converter = None
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.colors as mcolors
from matplotlib import rc as mplrc
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, LogNorm, SymLogNorm
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
# Plotting helpers
# ---------------------------------------------------------------------
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
        # Graceful fallback when colorspacious is not installed.
        rgb_colors = np.linspace(start_color, end_color, N)
    else:
        # Interpolate in CAM02-UCS for better perceptual uniformity.
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
    Compact 21-run sweep.

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
                            "at_H2O": 130.0,
                        }
                    },
                    "CO2": {
                        "release_temperatures_K": {
                            "pure": 60.0,
                            "at_H2O": 130.0,
                        }
                    },
                    "H2O": {
                        "release_temperature_K": 130.0,
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
                            "at_H2O": 170.0,
                        }
                    },
                    "CO2": {
                        "release_temperatures_K": {
                            "pure": 85.0,
                            "at_H2O": 170.0,
                        }
                    },
                    "H2O": {
                        "release_temperature_K": 170.0,
                    },
                }
            },
        },
        {
            "name": "release_different",
            "group": "release_temperature",
            "description": "Water guest < water release temperatures.",
            "overrides": {
                "volatiles": {
                    "CO": {
                        "release_temperatures_K": {
                            "pure": 30.0,
                            "at_CO2": 70.0,
                            "at_H2O": 130.0,
                        }
                    },
                    "CO2": {
                        "release_temperatures_K": {
                            "pure": 70.0,
                            "at_H2O": 130.0,
                        }
                    },
                    "H2O": {
                        "release_temperature_K": 150.0,
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
    """Load output-cadence-independent cumulative release diagnostics.

    New model outputs provide ``cum_dM_*`` columns in every snapshot.  For
    interval-integrated version-2 outputs, summing ``dM_*`` is equivalent.  Old
    version-1 outputs stored only the single phase step coinciding with each
    snapshot; those are retained for compatibility but explicitly marked as
    unreliable and should be regenerated for quantitative release radii.
    """
    if not snapshot_files:
        return {"have_release": False, "release_reliable": False, "release_mode": "missing"}

    first = read_snapshot(snapshot_files[0])
    final = read_snapshot(snapshot_files[-1])
    r = first["r_au"].to_numpy(dtype=float)
    out: Dict[str, Any] = {
        "have_release": False,
        "release_reliable": False,
        "release_mode": "missing",
        "r_au": r,
    }

    # Preferred path: direct lifetime integrals from the final snapshot.
    have_cumulative = any(f"cum_dM_{ch}" in final.columns for ch in RELEASE_CHANNELS)
    if have_cumulative:
        out["release_mode"] = "direct_cumulative_v2"
        out["release_reliable"] = True
        for channel in RELEASE_CHANNELS:
            colname = f"cum_dM_{channel}"
            out[f"dM_{channel}"] = (
                final[colname].to_numpy(dtype=float) if colname in final.columns else np.zeros_like(r)
            )
        out["dM_CO_gas"] = (
            final["cum_dM_CO_gas"].to_numpy(dtype=float)
            if "cum_dM_CO_gas" in final.columns else np.zeros_like(r)
        )
        out["dM_CO_gas_gain"] = (
            final["cum_dM_CO_gas_gain"].to_numpy(dtype=float)
            if "cum_dM_CO_gas_gain" in final.columns else np.zeros_like(r)
        )
    else:
        interval_v2 = "release_semantics_version" in final.columns or "release_interval_yr" in final.columns
        out["release_mode"] = "summed_intervals_v2" if interval_v2 else "legacy_snapshot_sample_v1"
        out["release_reliable"] = bool(interval_v2)
        for channel in RELEASE_CHANNELS:
            out[f"dM_{channel}"] = np.zeros_like(r)
        out["dM_CO_gas"] = np.zeros_like(r)
        out["dM_CO_gas_gain"] = np.zeros_like(r)
        for snap in snapshot_files:
            df = read_snapshot(snap)
            for channel in RELEASE_CHANNELS:
                colname = f"dM_{channel}"
                if colname in df.columns:
                    out[f"dM_{channel}"] += df[colname].to_numpy(dtype=float)
            if "dM_CO_gas" in df.columns:
                out["dM_CO_gas"] += df["dM_CO_gas"].to_numpy(dtype=float)
            if "dM_CO_gas_gain" in df.columns:
                out["dM_CO_gas_gain"] += df["dM_CO_gas_gain"].to_numpy(dtype=float)

    for channel in RELEASE_CHANNELS:
        if np.any(out[f"dM_{channel}"] > 0.0):
            out["have_release"] = True
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
    result["release_diagnostics_reliable"] = bool(rel.get("release_reliable", False))
    result["release_diagnostics_mode"] = rel.get("release_mode", "missing")

    initial_df = read_snapshot(snaps[0])
    initial_area = annulus_area_cm2(initial_df["r_au"].to_numpy(dtype=float))
    initial_channel_fields = {
        "CO_pure": ("CO_pure_ice_pebble", "CO_pure_ice_small"),
        "CO_at_CO2": ("CO_at_CO2_ice_pebble", "CO_at_CO2_ice_small"),
        "CO_at_H2O": ("CO_at_H2O_ice_pebble", "CO_at_H2O_ice_small"),
    }
    for channel in RELEASE_CHANNELS:
        for metric in ("R_peak", "R10", "R50", "R90", "Mtot_Mearth"):
            value = rel.get(f"{metric}_{channel}", np.nan)
            # Legacy v1 files sampled only the phase step coincident with each
            # output. Do not silently turn those cadence-dependent samples into
            # paper-facing release radii.
            if not result["release_diagnostics_reliable"]:
                value = np.nan
            result[f"{metric}_{channel}"] = value
        initial_mass = sum(
            float(np.nansum(initial_area * col(initial_df, field)))
            for field in initial_channel_fields[channel]
        )
        result[f"initial_Mearth_{channel}"] = initial_mass / MEARTH
        result[f"gross_release_over_initial_{channel}"] = (
            result[f"Mtot_Mearth_{channel}"] / max(initial_mass / MEARTH, EPS)
            if initial_mass > 0.0 else np.nan
        )

    result["total_CO_gross_release_Mearth"] = float(
        sum(result.get(f"Mtot_Mearth_{ch}", 0.0) for ch in RELEASE_CHANNELS)
    )
    # Backward-compatible alias.
    result["total_CO_release_Mearth"] = result["total_CO_gross_release_Mearth"]

    # Runtime-only diagnostics: boundary loss, capacity rejection, phase
    # cycling, positivity corrections, and numerical conservation.
    diag_path = outdir / "diagnostics.csv"
    if diag_path.exists():
        diag = pd.read_csv(diag_path)
        if not diag.empty:
            last = diag.iloc[-1]
            runtime_columns = [
                "retained_CO_fraction", "retained_CO2_fraction", "retained_H2O_fraction",
                "cum_boundary_inner_CO_mearth", "cum_boundary_outer_CO_mearth",
                "cum_boundary_inner_CO2_mearth", "cum_boundary_outer_CO2_mearth",
                "cum_boundary_inner_H2O_mearth", "cum_boundary_outer_H2O_mearth",
                "cum_boundary_inner_pebble_solids_mearth", "cum_boundary_outer_pebble_solids_mearth",
                "cum_boundary_inner_small_solids_mearth", "cum_boundary_outer_small_solids_mearth",
                "initial_capacity_excess_CO_at_CO2_mearth", "current_capacity_excess_CO_at_CO2_mearth", "current_capacity_active_cell_fraction_CO_at_CO2", "cum_capacity_excess_CO_at_CO2_mearth", "total_capacity_excess_CO_at_CO2_mearth",
                "initial_capacity_excess_CO_at_H2O_mearth", "current_capacity_excess_CO_at_H2O_mearth", "current_capacity_active_cell_fraction_CO_at_H2O", "cum_capacity_excess_CO_at_H2O_mearth", "total_capacity_excess_CO_at_H2O_mearth",
                "initial_capacity_excess_CO2_at_H2O_mearth", "current_capacity_excess_CO2_at_H2O_mearth", "current_capacity_active_cell_fraction_CO2_at_H2O", "cum_capacity_excess_CO2_at_H2O_mearth", "total_capacity_excess_CO2_at_H2O_mearth",
                "current_capacity_to_gas_CO_mearth", "current_capacity_to_gas_CO2_mearth", "total_capacity_to_gas_CO_mearth", "total_capacity_to_gas_CO2_mearth",
                "cum_phase_gas_gain_CO_mearth", "cum_phase_gas_loss_CO_mearth", "cum_phase_gas_net_CO_mearth", "phase_cycling_factor_CO",
                "cum_phase_gas_gain_CO2_mearth", "cum_phase_gas_loss_CO2_mearth", "cum_phase_gas_net_CO2_mearth", "phase_cycling_factor_CO2",
                "cum_phase_gas_gain_H2O_mearth", "cum_phase_gas_loss_H2O_mearth", "cum_phase_gas_net_H2O_mearth", "phase_cycling_factor_H2O",
                "mass_balance_residual_fraction_CO", "mass_balance_residual_fraction_CO2", "mass_balance_residual_fraction_H2O",
                "cum_clipped_added_CO_mearth", "cum_clipped_added_CO2_mearth", "cum_clipped_added_H2O_mearth",
                "phase_max_rel_error_CO", "phase_max_rel_error_CO2", "phase_max_rel_error_H2O",
                "mass_weighted_epsilon_pebble", "max_epsilon_pebble",
                "mass_weighted_rel_delta_v_gas_backreaction", "mass_weighted_rel_delta_v_pebble_backreaction",
                "dt_min_yr", "dt_mean_yr", "dt_max_yr", "n_steps",
            ]
            for c in runtime_columns:
                result[c] = float(last[c]) if c in diag.columns else np.nan
            result["runtime_diagnostics_available"] = True
            result["cum_boundary_total_CO_mearth"] = result.get("cum_boundary_inner_CO_mearth", 0.0) + result.get("cum_boundary_outer_CO_mearth", 0.0)
            result["cum_boundary_total_CO2_mearth"] = result.get("cum_boundary_inner_CO2_mearth", 0.0) + result.get("cum_boundary_outer_CO2_mearth", 0.0)
            result["cum_boundary_total_H2O_mearth"] = result.get("cum_boundary_inner_H2O_mearth", 0.0) + result.get("cum_boundary_outer_H2O_mearth", 0.0)
            result["total_capacity_excess_CO_mearth"] = result.get("total_capacity_excess_CO_at_CO2_mearth", 0.0) + result.get("total_capacity_excess_CO_at_H2O_mearth", 0.0)
            result["max_abs_mass_balance_residual_fraction"] = float(np.nanmax(np.abs([
                result.get("mass_balance_residual_fraction_CO", np.nan),
                result.get("mass_balance_residual_fraction_CO2", np.nan),
                result.get("mass_balance_residual_fraction_H2O", np.nan),
            ])))
            result["total_clipped_volatile_mearth"] = sum(result.get(f"cum_clipped_added_{sp}_mearth", 0.0) for sp in ("CO", "CO2", "H2O"))
        else:
            result["runtime_diagnostics_available"] = False
    else:
        result["runtime_diagnostics_available"] = False

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




# ---------------------------------------------------------------------
# Paper-ready plot helpers
# ---------------------------------------------------------------------
ICE_SUITE_NAMES = [
    "ice_pure_snowline",
    "ice_low_trap",
    "fiducial",
    "ice_co2_trap",
    "ice_h2o_trap",
    "fiducial_w_backreact",
    "uncapped",
]

RUN_LABELS_SINGLE = {
    "ice_pure_snowline": "pure snow-surface",
    "ice_low_trap": "low trapping",
    "fiducial": "fiducial",
    "ice_co2_trap": r"CO$_2$-rich trapping",
    "ice_h2o_trap": r"H$_2$O-rich trapping",
    "fiducial_w_backreact": "fiducial + backreaction",
    "uncapped": "uncapped",
    "st_pebble_0p003": r"$St_{\rm peb}=0.003$",
    "st_pebble_0p1": r"$St_{\rm peb}=0.1$",
    "alpha_1e_m4": r"$\alpha=\alpha_z=10^{-4}$",
    "alpha_1e_m2": r"$\alpha=\alpha_z=10^{-2}$",
    "cond_pebble_0p99": r"$w_{\rm small}^{\rm cond}=0.01$",
    "cond_equal_0p50": r"$w_{\rm small}^{\rm cond}=0.50$",
    "cond_small_0p90": r"$w_{\rm small}^{\rm cond}=0.90$",
    "vertical_Tatm_1p2": r"$T_{\rm atm}/T_{\rm mid}=1.2$",
    "vertical_Tatm_3p0": r"$T_{\rm atm}/T_{\rm mid}=3.0$",
    "vdiff_fiducial_off": "fiducial, no vapor diffusion",
    "vdiff_h2o_trap_off": r"H$_2$O-rich trapping, no vapor diffusion",
    "release_cool": "cool release",
    "release_warm": "warm release",
    "release_different": "different release",
}

RUN_LABELS_MULTILINE = {
    "ice_pure_snowline": "pure\nsnow-surface",
    "ice_low_trap": "low\ntrapping",
    "fiducial": "fiducial",
    "ice_co2_trap": "CO$_2$-rich\ntrapping",
    "ice_h2o_trap": "H$_2$O-rich\ntrapping",
    "fiducial_w_backreact": "fiducial\n+ backreaction",
    "uncapped": "uncapped",
    "vdiff_fiducial_off": "fiducial\nno vapor diffusion",
    "vdiff_h2o_trap_off": "H$_2$O-rich\nno vapor diffusion",
    "release_cool": "cool\nrelease",
    "release_warm": "warm\nrelease",
    "release_different": "different\nrelease",
}

CHANNEL_LABELS = {
    "CO_pure": "pure CO",
    "CO_at_CO2": r"CO@CO$_2$",
    "CO_at_H2O": r"CO@H$_2$O",
    "CO2_pure": r"pure CO$_2$",
    "CO2_at_H2O": r"CO$_{2}$@H$_2$O",
}

CHANNEL_COLORS = {
    "CO_pure": color_list[0],
    "CO_at_CO2": color_list[1],
    "CO_at_H2O": color_list[2],
    "CO2_pure": color_list[3],
    "CO2_at_H2O": color_list[4],
}


def pretty_run_label(run_name: str, multiline: bool = False) -> str:
    """Human-readable run labels for plot ticks and titles."""
    labels = RUN_LABELS_MULTILINE if multiline else RUN_LABELS_SINGLE
    return labels.get(run_name, str(run_name).replace("_", " "))


def apply_paper_axis_style(ax: plt.Axes) -> None:
    """Light, consistent styling for paper-facing summary plots."""
    ax.grid(True, alpha=0.25)
    ax.tick_params(direction="out")


def final_co_budget_table(
    df: pd.DataFrame,
    names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Build an area-integrated final CO budget table for selected runs.

    The returned fractions sum to unity:
        gas CO + pure CO ice + CO@CO2 + CO@H2O = total CO.
    """
    if names is None:
        names = ICE_SUITE_NAMES

    sub = ordered(df, names)
    rows: List[Dict[str, Any]] = []

    for _, row in sub.iterrows():
        run_name = row["run_name"]
        snaps = snapshot_paths(Path(row["output_dir"]))
        if not snaps:
            continue

        final_df = read_snapshot(snaps[-1])
        area = annulus_area_cm2(final_df["r_au"].to_numpy(dtype=float))

        co_gas = col(final_df, "CO_gas")
        co_pure = col(final_df, "CO_pure_ice_pebble") + col(final_df, "CO_pure_ice_small")
        co_at_co2 = col(final_df, "CO_at_CO2_ice_pebble") + col(final_df, "CO_at_CO2_ice_small")
        co_at_h2o = col(final_df, "CO_at_H2O_ice_pebble") + col(final_df, "CO_at_H2O_ice_small")

        m_gas = float(np.nansum(area * co_gas))
        m_pure = float(np.nansum(area * co_pure))
        m_at_co2 = float(np.nansum(area * co_at_co2))
        m_at_h2o = float(np.nansum(area * co_at_h2o))
        m_total = m_gas + m_pure + m_at_co2 + m_at_h2o
        denom = max(m_total, EPS)

        rows.append(
            {
                "run_name": run_name,
                "group": row.get("group", ""),
                "M_CO_gas": m_gas,
                "M_CO_pure_ice": m_pure,
                "M_CO_at_CO2": m_at_co2,
                "M_CO_at_H2O": m_at_h2o,
                "M_CO_total": m_total,
                "frac_CO_gas": m_gas / denom,
                "frac_CO_pure_ice": m_pure / denom,
                "frac_CO_at_CO2": m_at_co2 / denom,
                "frac_CO_at_H2O": m_at_h2o / denom,
                "frac_CO_hidden": (m_at_co2 + m_at_h2o) / denom,
                "frac_CO_solid_total": (m_pure + m_at_co2 + m_at_h2o) / denom,
            }
        )

    return pd.DataFrame(rows)


def plot_paper_ice_suite_summary(
    df: pd.DataFrame,
    analysis_dir: Path,
    names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Paper-facing two-panel summary:
      (a) final global CO budget,
      (b) matrix-dependent median gross-loss radii.
    """
    if names is None:
        names = ICE_SUITE_NAMES

    budget_df = final_co_budget_table(df, names=names)
    release_df = ordered(df, names)

    if budget_df.empty or release_df.empty:
        return budget_df

    analysis_dir = Path(analysis_dir)
    budget_df.to_csv(analysis_dir / "paper_final_co_budget.csv", index=False)

    # fig, axes = plt.subplots(
    #     1,
    #     2,
    #     figsize=(13.2, 5.0),
    #     gridspec_kw={"width_ratios": [1.25, 1.0]},
    #     constrained_layout=True,
    # )
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot A: stacked final CO budget.
    x = np.arange(len(budget_df))
    bottom = np.zeros(len(budget_df), dtype=float)
    budget_components = [
        ("frac_CO_gas", "gas CO", color_list[5]),
        ("frac_CO_pure_ice", "pure CO ice", CHANNEL_COLORS["CO_pure"]),
        ("frac_CO_at_CO2", r"CO@CO$_2$", CHANNEL_COLORS["CO_at_CO2"]),
        ("frac_CO_at_H2O", r"CO@H$_2$O", CHANNEL_COLORS["CO_at_H2O"]),
    ]
    for colname, label, color in budget_components:
        vals = budget_df[colname].to_numpy(dtype=float)
        ax.bar(x, vals, bottom=bottom, label=label, color=color, edgecolor="black", linewidth=0.35)
        bottom += vals

    ax.set_xticks(x)
    ax.set_xticklabels([pretty_run_label(n, multiline=True) for n in budget_df["run_name"]], rotation=0)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Fraction of total CO")
    ax.set_title("Final global CO budget")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=4,
        frameon=True,
    )
    apply_paper_axis_style(ax)
    fig.savefig(analysis_dir / "paper_ice_co_budget.png", dpi=240)
    plt.close(fig)

    # Plot B: median release radii.
    fig, ax = plt.subplots(figsize=(10, 6))
    release_df = ordered(release_df, budget_df["run_name"].tolist())
    x = np.arange(len(release_df))
    w = 0.25
    series = [
        ("R50_CO_pure", "CO_pure", -w),
        ("R50_CO_at_CO2", "CO_at_CO2", 0.0),
        ("R50_CO_at_H2O", "CO_at_H2O", w),
    ]
    for colname, channel, dx in series:
        ax.bar(
            x + dx,
            release_df[colname],
            width=w,
            label=CHANNEL_LABELS[channel],
            color=CHANNEL_COLORS[channel],
            edgecolor="black",
            linewidth=0.35,
        )

    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_run_label(n, multiline=True) for n in release_df["run_name"]], rotation=0)
    ax.set_ylabel(r"Median gross-loss radius, $R_{50}$ [au]")
    ax.set_title("Matrix-dependent gross CO-reservoir loss")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.14),
        ncol=4,
        frameon=True,
    )
    ax.grid(True, which="both", axis="y", alpha=0.25)

    fig.savefig(analysis_dir / "paper_ice_release_radius.png", dpi=240)
    plt.close(fig)

    return budget_df


def plot_paper_fiducial_release_profile(
    df: pd.DataFrame,
    analysis_dir: Path,
    run_name: str = "fiducial",
) -> None:
    """Paper-facing cumulative gross reservoir-loss profile for one representative run."""
    sub = df[df["run_name"] == run_name]
    if sub.empty:
        return

    row = sub.iloc[0]
    snaps = snapshot_paths(Path(row["output_dir"]))
    rel = cumulative_release(snaps)
    if not rel.get("have_release", False):
        return

    r = rel["r_au"]
    dlnr = dlnr_from_centers(r)

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for channel in RELEASE_CHANNELS:
        prof = rel[f"dM_{channel}"] / np.maximum(dlnr, EPS) / MEARTH
        ax.semilogx(
            r,
            prof,
            label=CHANNEL_LABELS[channel],
            color=CHANNEL_COLORS[channel],
            linewidth=2.0,
        )

    ax.set_xlabel("Radius [au]")
    ax.set_ylabel(r"$dM_{\rm CO,gross}^{\rm cum}/d\ln r$ [$M_\oplus$]")
    ax.set_title(f"Cumulative gross CO-reservoir loss in the {pretty_run_label(run_name)} run")
    ax.legend(frameon=True)
    ax.grid(True, which="both", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(analysis_dir) / f"paper_cumulative_release_{run_name}.png", dpi=240)
    plt.close(fig)


def plot_paper_sensitivity_summary(df: pd.DataFrame, analysis_dir: Path) -> None:
    """
    Paper-facing 2x2 sensitivity figure. The top row emphasizes dominant controls,
    while the bottom row shows robustness to secondary disk-structure choices.
    """
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.2), constrained_layout=True)
    ax_st, ax_cond = axes[0]
    ax_vert, ax_alpha = axes[1]

    # Pebble drift / Stokes number sensitivity.
    sub = include_fiducial(df, "stokes").sort_values("St_pebble")
    if not sub.empty:
        for colname, channel in [
            ("R50_CO_pure", "CO_pure"),
            ("R50_CO_at_CO2", "CO_at_CO2"),
            ("R50_CO_at_H2O", "CO_at_H2O"),
        ]:
            ax_st.plot(
                sub["St_pebble"],
                sub[colname],
                marker="o",
                linewidth=2.0,
                label=CHANNEL_LABELS[channel],
                color=CHANNEL_COLORS[channel],
            )
        ax_st.set_xscale("log")
        ax_st.set_yscale("log")
        ax_st.set_xlabel(r"Pebble Stokes number, $St_{\rm peb}$")
        ax_st.set_ylabel(r"$R_{50}$ [au]")
        ax_st.set_title("(a) Pebble coupling (drift + settling)")
        ax_st.legend(frameon=True)
        ax_st.grid(True, which="both", alpha=0.25)

    # Carrier history / recondensation sensitivity.
    sub = include_fiducial(df, "condensation").sort_values("cond_small")
    if not sub.empty:
        ax_cond.plot(
            sub["cond_small"],
            sub["global_C_over_O_pebble"],
            marker="o",
            linewidth=2.0,
            label="pebble volatile C/O",
            color=color_list[0],
        )
        ax_cond.plot(
            sub["cond_small"],
            sub["global_C_over_O_small"],
            marker="o",
            linewidth=2.0,
            label="small-grain volatile C/O",
            color=color_list[1],
        )
        ax_cond.plot(
            sub["cond_small"],
            sub["global_small_fraction_solid_volatile"],
            marker="o",
            linewidth=2.0,
            label="small-grain volatile fraction",
            color=color_list[2],
        )
        ax_cond.set_xlabel(r"Small-grain condensation weight, $w_{\rm small}^{\rm cond}$")
        ax_cond.set_ylabel("Global ratio / fraction")
        ax_cond.set_title("(b) Carrier assignment")
        ax_cond.legend(frameon=True)
        apply_paper_axis_style(ax_cond)

    # Vertical temperature sensitivity.
    sub = include_fiducial(df, "vertical").sort_values("T_atm_factor")
    if not sub.empty:
        ax_vert.plot(
            sub["T_atm_factor"],
            sub["final_hidden_CO_fraction"],
            marker="o",
            linewidth=2.0,
            label="matrix-associated CO fraction",
            color=color_list[0],
        )
        ax_vert.plot(
            sub["T_atm_factor"],
            sub["global_C_over_O_pebble"],
            marker="o",
            linewidth=2.0,
            label="pebble volatile C/O",
            color=color_list[1],
        )
        ax_vert.plot(
            sub["T_atm_factor"],
            sub["global_C_over_O_small"],
            marker="o",
            linewidth=2.0,
            label="small-grain volatile C/O",
            color=color_list[2],
        )
        ax_vert.set_xlabel(r"$T_{\rm atm}/T_{\rm mid}$")
        ax_vert.set_ylabel("Global ratio / fraction")
        ax_vert.set_title("(c) Vertical temperature")
        ax_vert.legend(frameon=True)
        apply_paper_axis_style(ax_vert)

    # Turbulence / vertical mixing sensitivity.
    sub = include_fiducial(df, "alpha").sort_values("alpha")
    if not sub.empty:
        ax_alpha.plot(
            sub["alpha"],
            sub["final_hidden_CO_fraction"],
            marker="o",
            linewidth=2.0,
            label="matrix-associated CO fraction",
            color=color_list[0],
        )
        ax_alpha.plot(
            sub["alpha"],
            sub["final_gas_CO_fraction"],
            marker="o",
            linewidth=2.0,
            label="gas CO fraction",
            color=color_list[1],
        )
        ax_alpha.plot(
            sub["alpha"],
            sub["global_small_fraction_solid_volatile"],
            marker="o",
            linewidth=2.0,
            label="small-grain volatile fraction",
            color=color_list[2],
        )
        ax_alpha.set_xscale("log")
        ax_alpha.set_xlabel(r"Turbulent parameter, $\alpha=\alpha_z$")
        ax_alpha.set_ylabel("Global fraction")
        ax_alpha.set_title("(d) Turbulent transport/mixing")
        ax_alpha.legend(frameon=True)
        ax_alpha.grid(True, which="both", alpha=0.25)

    fig.savefig(Path(analysis_dir) / "paper_sensitivity_summary.png", dpi=240)
    plt.close(fig)
    
def plot_runtime_diagnostic_summary(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Summarize retention, boundary delivery, capacity throughput, and cycling."""
    required = "runtime_diagnostics_available"
    if required not in df.columns:
        return
    sub = ordered(df[df[required] == True], ICE_SUITE_NAMES)
    if sub.empty:
        return
    x = np.arange(len(sub))
    labels = [pretty_run_label(name, multiline=True) for name in sub["run_name"]]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    width = 0.24
    for j, (c, label) in enumerate((("retained_CO_fraction", "CO"), ("retained_CO2_fraction", r"CO$_2$"), ("retained_H2O_fraction", r"H$_2$O"))):
        if c in sub.columns:
            axes[0, 0].bar(x + (j - 1) * width, sub[c], width=width, label=label)
    axes[0, 0].set_ylabel("Final / initial inventory"); axes[0, 0].set_title("Volatile retention"); axes[0, 0].legend(); axes[0, 0].set_ylim(bottom=0)

    inner = sub.get("cum_boundary_inner_CO_mearth", pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
    outer = sub.get("cum_boundary_outer_CO_mearth", pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
    axes[0, 1].bar(x, inner, label="inner boundary")
    axes[0, 1].bar(x, outer, bottom=inner, label="outer boundary")
    axes[0, 1].set_ylabel(r"Cumulative CO loss [$M_\oplus$]"); axes[0, 1].set_title("CO boundary transport"); axes[0, 1].legend()

    bottom = np.zeros(len(sub))
    cap_cols = (("total_capacity_excess_CO_at_CO2_mearth", r"CO@CO$_2$"), ("total_capacity_excess_CO_at_H2O_mearth", r"CO@H$_2$O"), ("total_capacity_excess_CO2_at_H2O_mearth", r"CO$_2$@H$_2$O"))
    for c, label in cap_cols:
        vals = sub.get(c, pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
        axes[1, 0].bar(x, vals, bottom=bottom, label=label); bottom += vals
    axes[1, 0].set_yscale("symlog", linthresh=1e-6); axes[1, 0].set_ylabel(r"Gross capacity rejection [$M_\oplus$]")
    axes[1, 0].set_title("Host-capacity limiter throughput"); axes[1, 0].legend(fontsize=8)

    gain = sub.get("cum_phase_gas_gain_CO_mearth", pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
    loss = sub.get("cum_phase_gas_loss_CO_mearth", pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
    axes[1, 1].bar(x - 0.18, gain, width=0.36, label="gas gain")
    axes[1, 1].bar(x + 0.18, loss, width=0.36, label="gas loss")
    axes[1, 1].set_yscale("symlog", linthresh=1e-6); axes[1, 1].set_ylabel(r"Cumulative CO phase exchange [$M_\oplus$]")
    axes[1, 1].set_title("Gross CO phase cycling"); axes[1, 1].legend()

    for ax in axes.flat:
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right"); ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout(); fig.savefig(analysis_dir / "runtime_process_summary_ice_suite.png", dpi=220); plt.close(fig)

    # Current (non-cumulative) cap activity is easier to interpret than gross
    # throughput when assessing where the final target is capacity limited.
    current_cols = [
        "current_capacity_excess_CO_at_CO2_mearth",
        "current_capacity_excess_CO_at_H2O_mearth",
        "current_capacity_excess_CO2_at_H2O_mearth",
    ]
    if any(c in sub.columns for c in current_cols):
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        width2 = 0.24
        labels2 = [r"CO@CO$_2$", r"CO@H$_2$O", r"CO$_2$@H$_2$O"]
        active_cols = [
            "current_capacity_active_cell_fraction_CO_at_CO2",
            "current_capacity_active_cell_fraction_CO_at_H2O",
            "current_capacity_active_cell_fraction_CO2_at_H2O",
        ]
        for j, (c, ac, label) in enumerate(zip(current_cols, active_cols, labels2)):
            vals = sub.get(c, pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
            active = sub.get(ac, pd.Series(np.zeros(len(sub)), index=sub.index)).to_numpy()
            axes[0].bar(x + (j - 1) * width2, vals, width=width2, label=label)
            axes[1].plot(x, active, marker="o", label=label)
        axes[0].set_ylabel("Current rejected target mass [Earth masses]")
        axes[0].set_title("Capacity-limited target at final time")
        axes[1].set_ylabel("Fraction of radial cells with active cap")
        axes[1].set_ylim(-0.02, 1.02); axes[1].set_title("Radial extent of active capacity limits")
        for ax in axes:
            ax.set_xticks(x); ax.set_xticklabels(labels, rotation=25, ha="right"); ax.grid(True, axis="y", alpha=0.25); ax.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(analysis_dir / "current_capacity_saturation_ice_suite.png", dpi=220); plt.close(fig)

    # All-run numerical quality check.
    allsub = df[df[required] == True].copy()
    if not allsub.empty and "max_abs_mass_balance_residual_fraction" in allsub.columns:
        allsub = allsub.sort_values("max_abs_mass_balance_residual_fraction")
        y = np.arange(len(allsub))
        fig, axes = plt.subplots(1, 2, figsize=(12, max(5.0, 0.34 * len(allsub))))
        axes[0].barh(y, np.maximum(np.abs(allsub["max_abs_mass_balance_residual_fraction"]), 1e-30))
        axes[0].set_xscale("log"); axes[0].set_xlabel("Maximum absolute mass-balance residual fraction"); axes[0].set_yticks(y); axes[0].set_yticklabels([pretty_run_label(n) for n in allsub["run_name"]], fontsize=8)
        if "total_clipped_volatile_mearth" in allsub.columns:
            axes[1].barh(y, np.maximum(allsub["total_clipped_volatile_mearth"], 1e-30))
            axes[1].set_xscale("log"); axes[1].set_xlabel(r"Cumulative clipped volatile mass [$M_\oplus$]"); axes[1].set_yticks(y); axes[1].set_yticklabels([])
        for ax in axes: ax.grid(True, axis="x", which="both", alpha=0.25)
        fig.tight_layout(); fig.savefig(analysis_dir / "sweep_numerical_quality.png", dpi=220); plt.close(fig)


def plot_ice_release(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Standalone paper-ready R50 gross-loss-radius summary for the ice suite."""
    names = ICE_SUITE_NAMES
    sub = ordered(df, names)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.25

    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.bar(x - w, sub["R50_CO_pure"], width=w, label=CHANNEL_LABELS["CO_pure"],
           color=CHANNEL_COLORS["CO_pure"], edgecolor="black", linewidth=0.35)
    ax.bar(x, sub["R50_CO_at_CO2"], width=w, label=CHANNEL_LABELS["CO_at_CO2"],
           color=CHANNEL_COLORS["CO_at_CO2"], edgecolor="black", linewidth=0.35)
    ax.bar(x + w, sub["R50_CO_at_H2O"], width=w, label=CHANNEL_LABELS["CO_at_H2O"],
           color=CHANNEL_COLORS["CO_at_H2O"], edgecolor="black", linewidth=0.35)
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty_run_label(n, multiline=True) for n in sub["run_name"]])
    ax.set_ylabel(r"Median gross-loss radius, $R_{50}$ [au]")
    ax.set_title("Ice matrix controls where CO reservoirs are depleted")
    ax.legend(frameon=True, ncol=3)
    ax.grid(True, which="both", axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(Path(analysis_dir) / "ice_suite_release_r50.png", dpi=240)
    plt.close(fig)


def plot_ice_partitioning(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["ice_pure_snowline", "ice_low_trap", "fiducial", "ice_co2_trap", "ice_h2o_trap", "fiducial_w_backreact", "uncapped"]
    sub = ordered(df, names)
    print(sub)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.35

    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, sub["final_hidden_CO_fraction"], width=w, label="hidden CO", color=color_list[0])
    plt.bar(x + w / 2, sub["final_gas_CO_fraction"], width=w, label="gas CO", color=color_list[1])
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel("Fraction of total CO")
    plt.ylim(0, 1)
    plt.title("Final CO partitioning")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    savefig(analysis_dir / "ice_suite_co_partitioning.png")


def plot_cumulative_release_profiles(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Write one cumulative-release profile per selected run for appendix diagnostics."""
    names = ICE_SUITE_NAMES
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
        fig, ax = plt.subplots(figsize=(9, 5))
        for channel in RELEASE_CHANNELS:
            prof = rel[f"dM_{channel}"] / np.maximum(dlnr, EPS) / MEARTH
            ax.semilogx(
                r,
                prof,
                label=CHANNEL_LABELS[channel],
                color=CHANNEL_COLORS[channel],
                linewidth=1.8,
            )
        ax.set_xlabel("Radius [au]")
        ax.set_ylabel(r"$dM_{\rm CO,gross}^{\rm cum}/d\ln r$ [$M_\oplus$]")
        ax.set_title(f"Cumulative gross CO-reservoir loss: {pretty_run_label(row['run_name'])}")
        ax.legend(frameon=True)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(analysis_dir) / f"cumulative_release_{row['run_name']}.png", dpi=220)
        plt.close(fig)


def plot_transport(df: pd.DataFrame, analysis_dir: Path) -> None:
    """Appendix-style individual sensitivity plots for transport/turbulence."""
    sub = include_fiducial(df, "stokes").sort_values("St_pebble")
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        for colname, channel in [
            ("R50_CO_pure", "CO_pure"),
            ("R50_CO_at_CO2", "CO_at_CO2"),
            ("R50_CO_at_H2O", "CO_at_H2O"),
        ]:
            ax.plot(
                sub["St_pebble"],
                sub[colname],
                marker="o",
                linewidth=2.0,
                label=CHANNEL_LABELS[channel],
                color=CHANNEL_COLORS[channel],
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"Pebble Stokes number, $St_{\rm peb}$")
        ax.set_ylabel(r"Median gross-loss radius, $R_{50}$ [au]")
        ax.set_title("Sensitivity to pebble drift")
        ax.legend(frameon=True)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(analysis_dir) / "stokes_sensitivity.png", dpi=220)
        plt.close(fig)

    sub = include_fiducial(df, "alpha").sort_values("alpha")
    if not sub.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(sub["alpha"], sub["final_hidden_CO_fraction"], marker="o",
                linewidth=2.0, label="matrix-associated CO fraction", color=color_list[0])
        ax.plot(sub["alpha"], sub["final_gas_CO_fraction"], marker="o",
                linewidth=2.0, label="gas CO fraction", color=color_list[1])
        ax.plot(sub["alpha"], sub["global_small_fraction_solid_volatile"], marker="o",
                linewidth=2.0, label="small-grain volatile fraction", color=color_list[2])
        ax.set_xscale("log")
        ax.set_xlabel(r"Turbulent parameter, $\alpha=\alpha_z$")
        ax.set_ylabel("Global fraction")
        ax.set_title("Sensitivity to turbulent transport and vertical mixing")
        ax.legend(frameon=True)
        ax.grid(True, which="both", alpha=0.25)
        fig.tight_layout()
        fig.savefig(Path(analysis_dir) / "alpha_sensitivity.png", dpi=220)
        plt.close(fig)


def plot_condensation(df: pd.DataFrame, analysis_dir: Path) -> None:
    sub = include_fiducial(df, "condensation").sort_values("cond_small")
    if sub.empty:
        return

    plt.figure(figsize=(8, 5))
    plt.plot(sub["cond_small"], sub["global_C_over_O_pebble"], marker="o", 
             label="pebble volatile C/O", color=color_list[0])
    plt.plot(sub["cond_small"], sub["global_C_over_O_small"], marker="o", 
             label="small-grain volatile C/O", color=color_list[1])
    plt.plot(sub["cond_small"], sub["global_small_fraction_solid_volatile"], marker="o", 
             label="small volatile fraction", color=color_list[2])
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
        "retained_CO_fraction",
        "cum_boundary_inner_CO_mearth",
        "cum_boundary_outer_CO_mearth",
        "total_capacity_excess_CO_mearth",
        "phase_cycling_factor_CO",
        "max_abs_mass_balance_residual_fraction",
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
        color=color_list[0],
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
        color=color_list[0],
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

    plt.plot([lo, hi], [lo, hi], linestyle="--", linewidth=1, color="gray")

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
            color=color_list[0],
        )
        plt.bar(
            x + w / 2,
            ice["classical_global_gas_CO_fraction"],
            width=w,
            label="classical freeze-out proxy",
            color=color_list[1],
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
            color=color_list[0],
        )
        plt.axvline(0.0, linewidth=1, color="gray")
        plt.xlabel(
            r"$M_{\rm CO,gas}^{\rm model} - "
            r"M_{\rm CO,gas}^{\rm classical}$ [M$_\oplus$]",
            color=color_list[1],
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
    plt.plot(sub["T_atm_factor"], sub["final_hidden_CO_fraction"], marker="o", label="matrix-associated CO fraction", color=color_list[0])
    plt.plot(sub["T_atm_factor"], sub["global_C_over_O_pebble"], marker="o", label="pebble volatile C/O", color=color_list[1])
    plt.plot(sub["T_atm_factor"], sub["global_C_over_O_small"], marker="o", label="small-grain volatile C/O", color=color_list[2])
    plt.xlabel(r"$T_{\rm atm}/T_{\rm mid}$")
    plt.ylabel("Ratio / fraction")
    plt.title("Vertical temperature sensitivity")
    plt.legend()
    plt.grid(True, alpha=0.3)
    savefig(analysis_dir / "vertical_temperature_sensitivity.png")
    
    
def plot_final_co_budget_stacked(
    df: pd.DataFrame,
    analysis_dir: Path,
    names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Make a stacked final CO budget figure.

    The stacked components are:
        - gas-phase CO
        - pure/unhidden CO ice
        - CO trapped in CO2-associated ice
        - CO trapped in H2O-associated ice

    These components sum to the total CO column by construction:
        CO_total = CO_gas + CO_pure_ice + CO_at_CO2_ice + CO_at_H2O_ice

    This is intended to replace/augment the older hidden-vs-gas bar plot,
    which did not show where the rest of the CO budget was stored.

    Parameters
    ----------
    df : pandas.DataFrame
        Completed sweep metrics table, usually `ok` inside analyze_sweep.
        Must contain run_name and output_dir.
    analysis_dir : pathlib.Path
        Directory where the plot and companion CSV are written.
    names : sequence of str, optional
        Run order to plot. If None, uses a compact ice-suite/default order.
    filename : str
        Name of the output PNG.

    Returns
    -------
    budget_df : pandas.DataFrame
        Table of global CO budget fractions and masses for the plotted runs.
    """

    if names is None:
        names = [
            "ice_pure_snowline",
            "ice_low_trap",
            "fiducial",
            "ice_co2_trap",
            "ice_h2o_trap",
            "fiducial_w_backreact",
            "uncapped",
        ]

    sub = ordered(df, names)
    if sub.empty:
        return pd.DataFrame()

    rows = []

    for _, row in sub.iterrows():
        run_name = row["run_name"]
        outdir = Path(row["output_dir"])
        snaps = snapshot_paths(outdir)

        if not snaps:
            continue

        final_df = read_snapshot(snaps[-1])
        r_au = final_df["r_au"].to_numpy(dtype=float)
        area = annulus_area_cm2(r_au)

        # CO components by reservoir.
        co_gas = col(final_df, "CO_gas")

        co_pure = (
            col(final_df, "CO_pure_ice_pebble")
            + col(final_df, "CO_pure_ice_small")
        )

        co_at_co2 = (
            col(final_df, "CO_at_CO2_ice_pebble")
            + col(final_df, "CO_at_CO2_ice_small")
        )

        co_at_h2o = (
            col(final_df, "CO_at_H2O_ice_pebble")
            + col(final_df, "CO_at_H2O_ice_small")
        )

        m_gas = float(np.nansum(area * co_gas))
        m_pure = float(np.nansum(area * co_pure))
        m_at_co2 = float(np.nansum(area * co_at_co2))
        m_at_h2o = float(np.nansum(area * co_at_h2o))

        m_total = m_gas + m_pure + m_at_co2 + m_at_h2o
        denom = max(m_total, EPS)

        rows.append(
            {
                "run_name": run_name,
                "group": row.get("group", ""),
                "M_CO_gas": m_gas,
                "M_CO_pure_ice": m_pure,
                "M_CO_at_CO2": m_at_co2,
                "M_CO_at_H2O": m_at_h2o,
                "M_CO_total": m_total,
                "frac_CO_gas": m_gas / denom,
                "frac_CO_pure_ice": m_pure / denom,
                "frac_CO_at_CO2": m_at_co2 / denom,
                "frac_CO_at_H2O": m_at_h2o / denom,
                "frac_CO_hidden": (m_at_co2 + m_at_h2o) / denom,
                "frac_CO_solid_total": (m_pure + m_at_co2 + m_at_h2o) / denom,
            }
        )

    budget_df = pd.DataFrame(rows)

    if budget_df.empty:
        return budget_df

    # Save the underlying numbers for captions/tables.
    analysis_dir = Path(analysis_dir)
    budget_df.to_csv(analysis_dir / "final_co_budget_stacked.csv", index=False)

    # Pretty display labels for the main figure.
    label_map = {
        "ice_pure_snowline": "pure\nsnowline",
        "ice_low_trap": "low\ntrap",
        "fiducial": "fiducial",
        "ice_co2_trap": "CO$_2$\ntrap",
        "ice_h2o_trap": "H$_2$O\ntrap",
        "fiducial_w_backreact": "fiducial\n+ backreaction",
        "uncapped": "uncapped",
    }

    x = np.arange(len(budget_df))
    xticklabels = [label_map.get(name, name) for name in budget_df["run_name"]]

    # Use a reservoir-consistent color scheme:
    # gas in gray, pure CO/CO@CO2/CO@H2O matching the release-channel colors.
    components = [
        ("frac_CO_gas", "gas CO", color_list[5]),
        ("frac_CO_pure_ice", "pure CO ice", color_list[0]),
        ("frac_CO_at_CO2", r"CO@CO$_2$", color_list[1]),
        ("frac_CO_at_H2O", r"CO@H$_2$O", color_list[2]),
    ]

    fig, ax = plt.subplots(figsize=(10, 5.2))

    bottom = np.zeros(len(budget_df), dtype=float)
    for colname, label, color in components:
        values = budget_df[colname].to_numpy(dtype=float)
        ax.bar(
            x,
            values,
            bottom=bottom,
            label=label,
            color=color,
            edgecolor="black",
            linewidth=0.4,
        )
        bottom += values

    ax.set_xticks(x)
    ax.set_xticklabels(xticklabels, rotation=25, ha="right")
    ax.set_ylabel("Fraction of total CO")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Final global CO budget")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
        ncol=4,
        frameon=True,
    )

    savefig(analysis_dir / "ice_suite_co_budget_stacked.png")

    return budget_df


def plot_vdiff(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["fiducial", "vdiff_fiducial_off", "ice_h2o_trap", "vdiff_h2o_trap_off"]
    sub = ordered(df, names)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.35
    plt.figure(figsize=(10, 5))
    plt.bar(x - w / 2, sub["final_hidden_CO_fraction"], width=w, label="hidden CO", color=color_list[0])
    plt.bar(x + w / 2, sub["final_gas_CO_fraction"], width=w, label="gas CO", color=color_list[1])
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel("Fraction of total CO")
    plt.ylim(0, 1)
    plt.title("Vapor diffusion sensitivity")
    plt.legend()
    plt.grid(True, axis="y", alpha=0.3)
    savefig(analysis_dir / "vapor_diffusion_summary.png")


def plot_release_temperature(df: pd.DataFrame, analysis_dir: Path) -> None:
    names = ["release_cool", "fiducial", "release_warm", "release_different"]
    sub = ordered(df, names)
    if sub.empty:
        return

    x = np.arange(len(sub))
    w = 0.25
    plt.figure(figsize=(9, 5))
    plt.bar(x - w, sub["R50_CO_pure"], width=w, label="pure CO", color=color_list[0])
    plt.bar(x, sub["R50_CO_at_CO2"], width=w, label=r"CO@CO$_{2}$", color=color_list[1])
    plt.bar(x + w, sub["R50_CO_at_H2O"], width=w, label=r"CO@H$_{2}$O", color=color_list[2])
    plt.yscale("log")
    plt.xticks(x, sub["run_name"], rotation=30, ha="right")
    plt.ylabel(r"Median release radius $R_{50}$ [au]")
    plt.title("Effective release-temperature sensitivity")
    plt.legend()
    plt.grid(True, which="both", axis="y", alpha=0.3)
    savefig(analysis_dir / "release_temperature_sensitivity.png")
    
# ---------------------------------------------------------------------
# Paper / appendix tables
# ---------------------------------------------------------------------
def _nested_get(d: Dict[str, Any], path: Sequence[str], default: Any = np.nan) -> Any:
    """Safely get a nested dictionary value."""
    cur = d
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _first_nested_get(
    d: Dict[str, Any],
    paths: Sequence[Sequence[str]],
    default: Any = np.nan,
) -> Any:
    """Return the first available nested value from a list of possible paths."""
    for path in paths:
        value = _nested_get(d, path, default=np.nan)
        if not _is_missing(value):
            return value
    return default


def _is_missing(x: Any) -> bool:
    if x is None:
        return True
    try:
        return bool(pd.isna(x))
    except Exception:
        return False


def _fmt_table_value(x: Any, sig: int = 3) -> str:
    """Compact formatter for table values."""
    if _is_missing(x):
        return "--"

    if isinstance(x, (bool, np.bool_)):
        return "yes" if bool(x) else "no"

    if isinstance(x, str):
        return x

    try:
        xf = float(x)
    except Exception:
        return str(x)

    if not np.isfinite(xf):
        return "--"

    if xf == 0.0:
        return "0"

    if 1.0e-3 <= abs(xf) < 1.0e4:
        return f"{xf:.{sig}g}"

    return f"{xf:.{sig}e}"


def _latex_escape_text(s: Any) -> str:
    """
    Escape ordinary text for LaTeX tables. Do not use this on strings that
    intentionally contain LaTeX math.
    """
    s = str(s)
    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in repl.items():
        s = s.replace(old, new)
    return s


TABLE_RUN_LABELS = {
    "ice_pure_snowline": "pure snow-surface",
    "ice_low_trap": "low trapping",
    "fiducial": "fiducial",
    "ice_co2_trap": r"CO$_2$-rich trapping",
    "ice_h2o_trap": r"H$_2$O-rich trapping",
    "fiducial_w_backreact": "fiducial + backreaction",
    "uncapped": "uncapped",
    "st_pebble_0p003": r"$St_{\rm peb}=0.003$",
    "st_pebble_0p1": r"$St_{\rm peb}=0.1$",
    "alpha_1e_m4": r"$\alpha=\alpha_z=10^{-4}$",
    "alpha_1e_m2": r"$\alpha=\alpha_z=10^{-2}$",
    "cond_pebble_0p99": r"$w_{\rm small}^{\rm cond}=0.01$",
    "cond_equal_0p50": r"$w_{\rm small}^{\rm cond}=0.50$",
    "cond_small_0p90": r"$w_{\rm small}^{\rm cond}=0.90$",
    "vertical_Tatm_1p2": r"$T_{\rm atm}/T_{\rm mid}=1.2$",
    "vertical_Tatm_3p0": r"$T_{\rm atm}/T_{\rm mid}=3.0$",
    "vdiff_fiducial_off": "fiducial, no vapor diffusion",
    "vdiff_h2o_trap_off": r"H$_2$O-rich trapping, no vapor diffusion",
    "release_cool": "cool release",
    "release_warm": "warm release",
    "release_different": "different release",
}


def _table_run_label(run_name: str) -> str:
    return TABLE_RUN_LABELS.get(str(run_name), str(run_name).replace("_", " "))


def _write_table_pair(
    table: pd.DataFrame,
    analysis_dir: Path,
    basename: str,
    latex_caption: Optional[str] = None,
    latex_label: Optional[str] = None,
) -> None:
    """
    Write a table as both CSV and LaTeX. Values are expected to already be
    formatted as strings if column-specific formatting is desired.
    """
    analysis_dir = Path(analysis_dir)
    table.to_csv(analysis_dir / f"{basename}.csv", index=False)

    latex = table.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        caption=latex_caption,
        label=latex_label,
    )

    with open(analysis_dir / f"{basename}.tex", "w", encoding="utf-8") as f:
        f.write(latex)


def _fiducial_yaml_from_manifest(manifest: pd.DataFrame) -> Optional[Dict[str, Any]]:
    """Load the fiducial YAML from the sweep manifest."""
    if "run_name" not in manifest.columns or "yaml_path" not in manifest.columns:
        return None

    fid = manifest[manifest["run_name"] == "fiducial"]
    if fid.empty:
        return None

    yaml_path = Path(str(fid.iloc[0]["yaml_path"]))
    if not yaml_path.exists():
        return None

    return read_yaml(yaml_path)


def write_fiducial_parameter_table(
    manifest: pd.DataFrame,
    analysis_dir: Path,
) -> pd.DataFrame:
    """
    Write a compact fiducial-parameter table for the main text or methods appendix.

    This reads the fiducial YAML listed in sweep_manifest.csv. Missing values are
    written as '--', so the function is robust to minor YAML-schema differences.
    """
    params = _fiducial_yaml_from_manifest(manifest)
    if params is None:
        return pd.DataFrame()

    rows = []

    def add(category: str, parameter: str, value: Any, description: str) -> None:
        rows.append(
            {
                "Category": category,
                "Parameter": parameter,
                "Fiducial value": _fmt_table_value(value),
                "Description": description,
            }
        )

    # Disk / grid.
    add(
        "Grid",
        r"$r_{\min}$ [au]",
        _first_nested_get(params, [["grid", "r_min_au"], ["disk", "r_min_au"], ["radial_grid", "r_min_au"]]),
        "Inner radial boundary.",
    )
    add(
        "Grid",
        r"$r_{\max}$ [au]",
        _first_nested_get(params, [["grid", "r_max_au"], ["disk", "r_max_au"], ["radial_grid", "r_max_au"]]),
        "Outer radial boundary.",
    )
    add(
        "Grid",
        r"$N_r$",
        _first_nested_get(params, [["grid", "n_r"], ["disk", "n_r"], ["radial_grid", "n_r"]]),
        "Number of radial cells.",
    )

    # Gas and turbulence.
    add("Gas", r"$\alpha$", _nested_get(params, ["gas", "alpha"]), "Radial turbulent transport parameter.")
    add("Gas", r"gas evolution", _nested_get(params, ["gas", "update_gas"], default=np.nan), "Whether the gas surface density is evolved.")
    add("Vertical", r"$\alpha_z$", _nested_get(params, ["vertical", "alpha_z"]), "Vertical stirring / settling parameter.")
    add(
        "Vertical",
        r"$T_{\rm atm}/T_{\rm mid}$",
        _nested_get(params, ["vertical", "temperature", "T_atm_factor"]),
        "Atmospheric temperature factor.",
    )
    add(
        "Vertical",
        r"$z_q/H_g$",
        _nested_get(params, ["vertical", "temperature", "zq_H"]),
        "Vertical temperature transition height.",
    )
    add(
        "Vertical",
        r"$p_T$",
        _nested_get(params, ["vertical", "temperature", "power"]),
        "Vertical temperature transition sharpness.",
    )
    add(
        "Shielding",
        r"$A_{V,\rm crit}$",
        _nested_get(params, ["vertical", "shielding", "Av_crit"]),
        "Critical visual extinction for shielding.",
    )

    # Dust.
    add("Dust", r"$St_{\rm peb}$", _nested_get(params, ["dust", "carriers", "pebble", "St"]), "Pebble Stokes number.")
    add("Dust", r"$St_{\rm small}$", _nested_get(params, ["dust", "carriers", "small", "St"]), "Small-grain Stokes number.")
    add(
        "Dust",
        r"$w_{\rm peb}^{\rm cond}$",
        _nested_get(params, ["dust", "volatile_carrier_fractions", "pebble"]),
        "Pebble share of newly condensed volatile ice.",
    )
    add(
        "Dust",
        r"$w_{\rm small}^{\rm cond}$",
        _nested_get(params, ["dust", "volatile_carrier_fractions", "small"]),
        "Small-grain share of newly condensed volatile ice.",
    )
    add(
        "Dust",
        "backreaction",
        _first_nested_get(params, [["dust", "backreaction", "enabled"], ["dust", "backreaction"]]),
        "Whether dust backreaction modifies carrier/gas velocities.",
    )

    # CO partition and release.
    add("CO", r"$f_{\rm pure}$", _nested_get(params, ["volatiles", "CO", "solid_fractions", "pure"]), "Pure CO-ice fraction.")
    add("CO", r"$f_{\rm CO@CO_2}$", _nested_get(params, ["volatiles", "CO", "solid_fractions", "at_CO2"]), r"CO fraction following CO$_2$-associated release.")
    add("CO", r"$f_{\rm CO@H_2O}$", _nested_get(params, ["volatiles", "CO", "solid_fractions", "at_H2O"]), r"CO fraction following H$_2$O-associated release.")
    add("CO", r"$T_{\rm rel,pure}$ [K]", _nested_get(params, ["volatiles", "CO", "release_temperatures_K", "pure"]), "Pure CO release temperature.")
    add("CO", r"$T_{\rm rel,CO@CO_2}$ [K]", _nested_get(params, ["volatiles", "CO", "release_temperatures_K", "at_CO2"]), r"CO@CO$_2$ release temperature.")
    add("CO", r"$T_{\rm rel,CO@H_2O}$ [K]", _nested_get(params, ["volatiles", "CO", "release_temperatures_K", "at_H2O"]), r"CO@H$_2$O release temperature.")

    # CO2 and H2O.
    add(r"CO$_2$", r"$f_{\rm pure}$", _nested_get(params, ["volatiles", "CO2", "solid_fractions", "pure"]), r"Pure CO$_2$-ice fraction.")
    add(r"CO$_2$", r"$f_{\rm CO_2@H_2O}$", _nested_get(params, ["volatiles", "CO2", "solid_fractions", "at_H2O"]), r"CO$_2$ fraction following H$_2$O-associated release.")
    add(r"H$_2$O", r"$T_{\rm rel,H_2O}$ [K]", _nested_get(params, ["volatiles", "H2O", "release_temperature_K"]), r"H$_2$O release temperature.")

    # Capacity.
    cap = _nested_get(params, ["volatiles", "trapping_capacity"], default={})
    add("Capacity", "enabled", _nested_get(cap, ["enabled"]), "Whether host-capacity limits are applied.")
    add("Capacity", "excess destination", _nested_get(cap, ["excess_destination"]), "Reservoir receiving over-capacity guest volatile.")
    add("Capacity", r"$q_{\rm CO|CO_2}$", _nested_get(cap, ["CO_at_CO2", "max_guest_per_host_mol"]), r"Maximum CO/CO$_2$ guest-host molecular ratio.")
    add("Capacity", r"$q_{\rm CO|H_2O}$", _nested_get(cap, ["CO_at_H2O", "max_guest_per_host_mol"]), r"Maximum CO/H$_2$O guest-host molecular ratio.")
    add("Capacity", r"$q_{\rm CO_2|H_2O}$", _nested_get(cap, ["CO2_at_H2O", "max_guest_per_host_mol"]), r"Maximum CO$_2$/H$_2$O guest-host molecular ratio.")

    table = pd.DataFrame(rows)
    _write_table_pair(
        table,
        analysis_dir,
        "table_fiducial_parameters",
        latex_caption="Fiducial model parameters.",
        latex_label="tab:fiducial_parameters",
    )
    return table


def write_main_results_table(
    metrics: pd.DataFrame,
    analysis_dir: Path,
    names: Optional[Sequence[str]] = None,
) -> pd.DataFrame:
    """
    Write a compact main-text results table for the core ice suite.
    """
    if names is None:
        names = [
            "ice_pure_snowline",
            "ice_low_trap",
            "fiducial",
            "ice_co2_trap",
            "ice_h2o_trap",
            "fiducial_w_backreact",
            "uncapped",
        ]

    sub = ordered(metrics, names)
    if sub.empty:
        return pd.DataFrame()

    rows = []
    for _, row in sub.iterrows():
        rows.append(
            {
                "Run": _table_run_label(row["run_name"]),
                r"$f_{\rm CO,matrix}$": _fmt_table_value(row.get("final_hidden_CO_fraction")),
                r"$f_{\rm CO,gas}$": _fmt_table_value(row.get("final_gas_CO_fraction")),
                r"$R_{50}^{\rm pure}$ [au]": _fmt_table_value(row.get("R50_CO_pure")),
                r"$R_{50}^{\rm CO@CO_2}$ [au]": _fmt_table_value(row.get("R50_CO_at_CO2")),
                r"$R_{50}^{\rm CO@H_2O}$ [au]": _fmt_table_value(row.get("R50_CO_at_H2O")),
                "Pebble volatile C/O": _fmt_table_value(row.get("global_C_over_O_pebble")),
                "Small-grain volatile C/O": _fmt_table_value(row.get("global_C_over_O_small")),
            }
        )

    table = pd.DataFrame(rows)
    _write_table_pair(
        table,
        analysis_dir,
        "table_main_results",
        latex_caption="Summary of final CO partitioning and median gross reservoir-loss radii for the main model suite.",
        latex_label="tab:main_results",
    )
    return table


def write_sweep_definition_table(
    manifest: pd.DataFrame,
    analysis_dir: Path,
    fiducial_name: str = "fiducial",
) -> pd.DataFrame:
    """
    Write a compact appendix table describing how each sweep run differs from
    the fiducial model.

    The function compares each manifest row against the fiducial row and writes
    only the changed parameters.
    """
    if manifest.empty or "run_name" not in manifest.columns:
        return pd.DataFrame()

    fid = manifest[manifest["run_name"] == fiducial_name]
    fid_row = fid.iloc[0] if not fid.empty else None

    compare_cols = [
        ("CO_pure", r"$f_{\rm CO,pure}$"),
        ("CO_at_CO2", r"$f_{\rm CO@CO_2}$"),
        ("CO_at_H2O", r"$f_{\rm CO@H_2O}$"),
        ("CO2_pure", r"$f_{\rm CO_2,pure}$"),
        ("CO2_at_H2O", r"$f_{\rm CO_2@H_2O}$"),
        ("St_pebble", r"$St_{\rm peb}$"),
        ("St_small", r"$St_{\rm small}$"),
        ("alpha", r"$\alpha$"),
        ("alpha_z", r"$\alpha_z$"),
        ("cond_pebble", r"$w_{\rm peb}^{\rm cond}$"),
        ("cond_small", r"$w_{\rm small}^{\rm cond}$"),
        ("T_atm_factor", r"$T_{\rm atm}/T_{\rm mid}$"),
        ("vapor_diffusion", "vapor diffusion"),
        ("Trel_CO_pure", r"$T_{\rm rel,CO,pure}$"),
        ("Trel_CO_at_CO2", r"$T_{\rm rel,CO@CO_2}$"),
        ("Trel_CO_at_H2O", r"$T_{\rm rel,CO@H_2O}$"),
    ]

    rows = []
    for _, row in manifest.iterrows():
        run_name = row["run_name"]

        changed = []
        if fid_row is None or run_name == fiducial_name:
            changed = ["fiducial values"]
        else:
            for colname, label in compare_cols:
                if colname not in manifest.columns:
                    continue

                val = row.get(colname)
                fid_val = fid_row.get(colname)

                if _is_missing(val) and _is_missing(fid_val):
                    continue

                # Use allclose for numeric values and direct comparison otherwise.
                differs = False
                try:
                    differs = not np.isclose(float(val), float(fid_val), rtol=1.0e-12, atol=1.0e-300)
                except Exception:
                    differs = str(val) != str(fid_val)

                if differs:
                    changed.append(f"{label} = {_fmt_table_value(val)}")

        if not changed:
            changed = ["metadata/control change"]

        rows.append(
            {
                "Run": _table_run_label(run_name),
                "Group": _latex_escape_text(row.get("group", "")),
                "Changed parameters": "; ".join(changed),
                "Purpose": _latex_escape_text(row.get("description", "")),
            }
        )

    table = pd.DataFrame(rows)
    _write_table_pair(
        table,
        analysis_dir,
        "table_sweep_definitions",
        latex_caption="Definition of the parameter-sweep models. Changed parameters are listed relative to the fiducial model.",
        latex_label="tab:sweep_definitions",
    )
    return table


def write_capacity_assumptions_table(
    manifest: pd.DataFrame,
    analysis_dir: Path,
) -> pd.DataFrame:
    """
    Write a compact table describing the host-capacity assumptions in the
    fiducial model.
    """
    params = _fiducial_yaml_from_manifest(manifest)
    if params is None:
        return pd.DataFrame()

    cap = _nested_get(params, ["volatiles", "trapping_capacity"], default={})
    if not isinstance(cap, dict):
        return pd.DataFrame()

    excess_destination = _nested_get(cap, ["excess_destination"], default="gas")

    co_co2_host_mode = _nested_get(cap, ["CO_at_CO2", "host_mode"], default="pure_CO2_ice")
    if co_co2_host_mode == "pure_CO2_ice":
        co_co2_host = r"pure CO$_2$ ice"
    elif co_co2_host_mode == "total_CO2_ice":
        co_co2_host = r"all CO$_2$-bearing ice"
    else:
        co_co2_host = _latex_escape_text(co_co2_host_mode)

    rows = [
        {
            "Guest reservoir": r"CO@CO$_2$",
            "Host reservoir": co_co2_host,
            "Molecular cap": _fmt_table_value(_nested_get(cap, ["CO_at_CO2", "max_guest_per_host_mol"])),
            "Excess treatment": _latex_escape_text(excess_destination),
        },
        {
            "Guest reservoir": r"CO@H$_2$O",
            "Host reservoir": r"H$_2$O ice",
            "Molecular cap": _fmt_table_value(_nested_get(cap, ["CO_at_H2O", "max_guest_per_host_mol"])),
            "Excess treatment": _latex_escape_text(excess_destination),
        },
        {
            "Guest reservoir": r"CO$_2$@H$_2$O",
            "Host reservoir": r"H$_2$O ice",
            "Molecular cap": _fmt_table_value(_nested_get(cap, ["CO2_at_H2O", "max_guest_per_host_mol"])),
            "Excess treatment": _latex_escape_text(excess_destination),
        },
    ]

    total_h2o_cap = _nested_get(cap, ["H2O_total_guest_capacity", "max_total_guest_per_host_mol"], default=np.nan)
    if not _is_missing(total_h2o_cap):
        rows.append(
            {
                "Guest reservoir": r"CO@H$_2$O + CO$_2$@H$_2$O",
                "Host reservoir": r"H$_2$O ice",
                "Molecular cap": _fmt_table_value(total_h2o_cap),
                "Excess treatment": _latex_escape_text(excess_destination),
            }
        )

    table = pd.DataFrame(rows)
    _write_table_pair(
        table,
        analysis_dir,
        "table_capacity_assumptions",
        latex_caption="Host-capacity assumptions used in the fiducial capacity-limited model.",
        latex_label="tab:capacity_assumptions",
    )
    return table


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
            *([f"- Retained CO fraction: {row['retained_CO_fraction']:.4f}"] if pd.notna(row.get('retained_CO_fraction', np.nan)) else []),
            *([f"- Cumulative inner-boundary CO delivery: {row['cum_boundary_inner_CO_mearth']:.3e} M_Earth"] if pd.notna(row.get('cum_boundary_inner_CO_mearth', np.nan)) else []),
            *([f"- CO phase cycling factor: {row['phase_cycling_factor_CO']:.3g}"] if pd.notna(row.get('phase_cycling_factor_CO', np.nan)) else []),
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

    if "release_diagnostics_reliable" in ok.columns:
        n_legacy = int((~ok["release_diagnostics_reliable"].fillna(False)).sum())
        if n_legacy:
            print(
                f"WARNING: {n_legacy} completed run(s) use legacy snapshot-sampled "
                "release diagnostics. Their release masses and radii are set to NaN; "
                "rerun them with mixed_ice_transport_1p1d_fixed.py."
            )

    ### for paper ####
    plot_paper_ice_suite_summary(ok, analysis_dir)
    plot_paper_fiducial_release_profile(ok, analysis_dir)
    plot_paper_sensitivity_summary(ok, analysis_dir)
    plot_final_co_budget_stacked(ok, analysis_dir)
    plot_ice_release(ok, analysis_dir)

    ### runtime/process diagnostics ####
    plot_runtime_diagnostic_summary(ok, analysis_dir)

    ### additional diagnostics / appendix ####
    plot_ice_partitioning(ok, analysis_dir)
    plot_cumulative_release_profiles(ok, analysis_dir)
    plot_transport(ok, analysis_dir)
    plot_condensation(ok, analysis_dir)
    plot_vertical(ok, analysis_dir)
    plot_vdiff(ok, analysis_dir)
    plot_release_temperature(ok, analysis_dir)

    ### exploratory / appendix only: bookkeeping-sensitive freeze-out proxy ####
    plot_freezeout(ok, analysis_dir)


    #### tables ####
    write_key_results(ok, analysis_dir)
    
    ### for paper tables ####
    write_fiducial_parameter_table(manifest, analysis_dir)
    write_main_results_table(ok, analysis_dir)

    ### appendix tables ####
    write_sweep_definition_table(manifest, analysis_dir)
    write_capacity_assumptions_table(manifest, analysis_dir)

    print(f"Analyzed {len(ok)} completed runs out of {len(metrics)} total.")
    print(f"Wrote metrics: {metrics_path}")
    print(f"Wrote plots/tables to: {analysis_dir}")


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate/analyze a 21-run mixed-ice sweep.")
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
