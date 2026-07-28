#!/usr/bin/env python3
"""Validate bookkeeping and basic consistency of a mixed-ice model output.

Usage
-----
    python validate_mixed_ice_outputs.py OUTPUT_DIR

The checks are intentionally numerical/bookkeeping checks, not tests of the
physical assumptions.  The script exits non-zero if a required check fails.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

EPS = 1.0e-300

RELEASE_CHANNELS = (
    "CO_pure",
    "CO_at_CO2",
    "CO_at_H2O",
    "CO2_pure",
    "CO2_at_H2O",
    "H2O_pure",
)

SOLID_FIELDS = (
    "ref_solid_pebble",
    "H2O_ice_pebble",
    "CO2_pure_ice_pebble",
    "CO2_at_H2O_ice_pebble",
    "CO_pure_ice_pebble",
    "CO_at_CO2_ice_pebble",
    "CO_at_H2O_ice_pebble",
    "ref_solid_small",
    "H2O_ice_small",
    "CO2_pure_ice_small",
    "CO2_at_H2O_ice_small",
    "CO_pure_ice_small",
    "CO_at_CO2_ice_small",
    "CO_at_H2O_ice_small",
)
VAPOR_FIELDS = ("CO_gas", "CO2_gas", "H2O_gas")


def snapshots(output_dir: Path) -> List[Path]:
    return sorted((output_dir / "snapshots").glob("snapshot_*.csv"))


def read(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def allclose_report(a: np.ndarray, b: np.ndarray, rtol: float = 2e-12) -> Dict[str, float | bool]:
    scale = max(float(np.nanmax(np.abs(a))), float(np.nanmax(np.abs(b))), 1.0)
    max_abs = float(np.nanmax(np.abs(a - b)))
    return {
        "ok": bool(np.allclose(a, b, rtol=rtol, atol=rtol * scale)),
        "max_abs_error": max_abs,
        "relative_to_peak": max_abs / scale,
    }


def validate(output_dir: Path) -> Dict[str, object]:
    files = snapshots(output_dir)
    if not files:
        raise FileNotFoundError(f"No snapshots found in {output_dir / 'snapshots'}")

    frames = [read(path) for path in files]
    first, final = frames[0], frames[-1]
    checks: Dict[str, object] = {
        "output_dir": str(output_dir),
        "n_snapshots": len(files),
        "final_time_yr": float(final["time_yr"].iloc[0]) if "time_yr" in final else None,
    }
    failures: List[str] = []

    # Basic finite/non-negative state check.
    state_fields = [field for field in VAPOR_FIELDS + SOLID_FIELDS if field in final.columns]
    nonfinite = {field: int((~np.isfinite(final[field].to_numpy(dtype=float))).sum()) for field in state_fields}
    negative = {field: float(final[field].min()) for field in state_fields if float(final[field].min()) < -1e-30}
    checks["nonfinite_counts"] = nonfinite
    checks["negative_fields"] = negative
    if any(nonfinite.values()):
        failures.append("non-finite state values")
    if negative:
        failures.append("negative state values")

    # Local CO partition identity.
    required_co = [
        "CO_gas",
        "CO_pure_ice_pebble", "CO_pure_ice_small",
        "CO_at_CO2_ice_pebble", "CO_at_CO2_ice_small",
        "CO_at_H2O_ice_pebble", "CO_at_H2O_ice_small",
    ]
    if all(column in final.columns for column in required_co):
        total = sum(final[column].to_numpy(dtype=float) for column in required_co)
        parts = {
            "gas": final["CO_gas"].to_numpy(dtype=float),
            "pure": final["CO_pure_ice_pebble"].to_numpy(dtype=float) + final["CO_pure_ice_small"].to_numpy(dtype=float),
            "CO@CO2": final["CO_at_CO2_ice_pebble"].to_numpy(dtype=float) + final["CO_at_CO2_ice_small"].to_numpy(dtype=float),
            "CO@H2O": final["CO_at_H2O_ice_pebble"].to_numpy(dtype=float) + final["CO_at_H2O_ice_small"].to_numpy(dtype=float),
        }
        frac_sum = sum(value / np.maximum(total, EPS) for value in parts.values())
        mask = total > 1e-14 * max(float(np.nanmax(total)), EPS)
        error = float(np.nanmax(np.abs(frac_sum[mask] - 1.0))) if np.any(mask) else 0.0
        checks["max_local_CO_fraction_sum_error"] = error
        if error > 1e-10:
            failures.append("local CO fractions do not sum to one")

    # Release semantics and interval/cumulative consistency.
    version = float(final["release_semantics_version"].iloc[0]) if "release_semantics_version" in final else 1.0
    checks["release_semantics_version"] = version
    if version < 2.0:
        failures.append("legacy release bookkeeping; rerun with the fixed simulation script")
    else:
        release_checks: Dict[str, object] = {}
        for channel in RELEASE_CHANNELS:
            interval_col = f"dM_{channel}"
            cumulative_col = f"cum_dM_{channel}"
            if cumulative_col not in final.columns:
                continue
            summed = np.zeros(len(final), dtype=float)
            previous = np.full(len(final), -np.inf)
            monotonic = True
            for frame in frames:
                if interval_col in frame.columns:
                    summed += frame[interval_col].to_numpy(dtype=float)
                current = frame[cumulative_col].to_numpy(dtype=float)
                if np.any(current + 1e-12 * np.maximum(np.abs(current), 1.0) < previous):
                    monotonic = False
                previous = current
            direct = final[cumulative_col].to_numpy(dtype=float)
            report = allclose_report(summed, direct)
            report["monotonic"] = monotonic
            release_checks[channel] = report
            if not report["ok"]:
                failures.append(f"interval/cumulative mismatch for {channel}")
            if not monotonic:
                failures.append(f"non-monotonic cumulative gross release for {channel}")
        checks["release_checks"] = release_checks

    # Diagnostics header uniqueness.
    diag_path = output_dir / "diagnostics.csv"
    if diag_path.exists():
        header = diag_path.read_text(encoding="utf-8").splitlines()[0].split(",")
        duplicates = sorted({name for name in header if header.count(name) > 1})
        checks["diagnostics_duplicate_headers"] = duplicates
        if duplicates:
            failures.append("duplicate diagnostics columns")

    checks["failures"] = failures
    checks["passed"] = not failures
    return checks


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate mixed-ice output bookkeeping.")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--json", type=Path, default=None, help="Optional path for a JSON report.")
    args = parser.parse_args()

    report = validate(args.output_dir.expanduser().resolve())
    print(json.dumps(report, indent=2))
    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
