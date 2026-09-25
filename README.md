# MIST
## Mixed-Ice Snowsurface Transport

<p align="center">
  <img src="mist_logo.png" alt="MIST logo" width="300">
</p>

**MIST** is a 1+1D model for the transport, sequestration, and release of volatile ices in protoplanetary disks.

The model evolves radial surface densities for gas and two solid carrier populations—settled pebbles and vertically extended small grains—while using a prescribed vertical disk structure to determine temperature-, shielding-, and carrier-dependent volatile survival. MIST is designed to explore how mixed ice matrices can allow volatile species to survive interior to their pure-ice snowlines and be released at multiple host-dependent fronts.

## What MIST models

MIST currently tracks CO, $CO_{2}$, and H$_2$O in gaseous, pure-ice, and matrix-associated reservoirs.

For CO:
- gas-phase CO
- pure CO ice
- CO sequestered in CO$_2$-rich ice (`CO@CO2`)
- CO sequestered in H$_2$O-rich ice (`CO@H2O`)

For CO$_2$:
- gas-phase CO$_2$
- pure CO$_2$ ice
- CO$_2$ sequestered in H$_2$O-rich ice (`CO2@H2O`)

For H$_2$O:
- gas-phase H$_2$O
- H$_2$O ice

Each solid reservoir is carried independently by:
- **pebbles** — settled, faster-drifting, mass-dominant solids
- **small grains** — more vertically extended and nearly gas-coupled

The model can additionally include:
- turbulent radial diffusion of solids and vapor
- prescribed gas evolution
- vertical temperature and shielding structure
- carrier-dependent dust scale heights
- finite host-matrix trapping capacities
- dust backreaction on gas and pebble velocities
- reversible or irreversible matrix sequestration
- optional reconstructed 2D `(r,z)` output

## The 1+1D approximation

The dynamically evolved quantities are radial surface densities,

$$
\Sigma_i(r,t),
$$

so radial transport is one-dimensional and Eulerian.

At each radius, MIST constructs a prescribed vertical disk column and calculates:
- $T(r,z)$
- shielding $A_V(r,z)$
- carrier-dependent scale heights
- matrix-dependent volatile survival probabilities
- snow/release surfaces

These vertical calculations are used to determine vertically averaged phase source terms for the 1D transport model.

The optional 2D outputs reconstruct the expected vertical distribution of each reservoir for diagnostics and visualization. They are **not** the result of independently evolved 2D transport.

## Installation

MIST is currently a script-based research code rather than an installable Python package.

Python **3.10+** is recommended.

Clone the repository:

```bash
git clone https://github.com/j-p-terry/mist.git
cd mist
```

A minimal virtual environment can be created with:

```bash
python -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install numpy pandas matplotlib pyyaml colorspacious
```

`colorspacious` is optional and is used only for improved perceptually uniform plotting.

## Quick start

An example parameter file is provided at:

```text
scripts/example_mist_params.yaml
```

Run a simulation from the repository root with:

```bash
python scripts/mixed_ice_transport_1p1d.py scripts/example_mist_params.yaml
```

The output directory is controlled by the YAML file, e.g.

```yaml
simulation:
  output_dir: mixed_ice_1p1d_outputs_example
```

## Configuration

Most model behavior is controlled through the YAML parameter file. Major configuration groups include:

```yaml
simulation:
grid:
star:
gas:
dust:
vertical:
volatiles:
initial_conditions:
output:
numerics:
```

Important options include:

```yaml
dust:
  carriers:
    pebble:
      St: 0.03
    small:
      St: 1.0e-4

  volatile_carrier_fractions:
    pebble: 0.90
    small: 0.10

  phase_partition:
    preserve_carrier_history: true
    allow_resequestration: true

  backreaction:
    enabled: false
```

and the volatile reservoir definitions:

```yaml
volatiles:
  CO:
    solid_fractions:
      pure: 0.40
      at_CO2: 0.40
      at_H2O: 0.20

    release_temperatures_K:
      pure: 25.0
      at_CO2: 70.0
      at_H2O: 150.0
```

Finite host capacity can be configured separately:

```yaml
volatiles:
  trapping_capacity:
    enabled: true

    CO_at_CO2:
      max_guest_per_host_mol: 0.5

    CO_at_H2O:
      max_guest_per_host_mol: 0.25

    CO2_at_H2O:
      max_guest_per_host_mol: 0.5
```

See `scripts/example_mist_params.yaml` for the full parameter set.

## Outputs

A standard run produces:

```text
<output_dir>/
├── resolved_params.yaml
├── diagnostics.csv
├── snapshots/
│   ├── snapshot_000000.csv
│   ├── snapshot_000001.csv
│   └── ...
└── snapshots_2d/
    ├── snapshot2d_000000.npz
    └── ...
```

The 1D snapshots contain the radial gas, pure-ice, and matrix-associated reservoirs, carrier diagnostics, snow/release-surface information, and phase-release bookkeeping.

Release diagnostics distinguish between:
- **gross positive loss from an ice reservoir**
- **net gas-phase source terms**

These are not generally identical because released material can recondense or pass through more than one reservoir.

The optional 2D `.npz` outputs contain reconstructed vertical quantities such as temperature, shielding, survival probabilities, release surfaces, and reconstructed gas/ice distributions.

## Analyze a run

For the latest snapshot:

```bash
python scripts/analyse_run.py mixed_ice_1p1d_outputs_example --snap latest
```

Other snapshot selectors include:

```text
first
quarter
middle
3quarter
latest
```

or an integer snapshot index.

To skip 2D analysis:

```bash
python scripts/analyse_run.py mixed_ice_1p1d_outputs_example --snap latest --skip-2d
```

Plots are written by default to:

```text
<output_dir>/analysis_plots/snap_<selection>/
```

## Validate a run

A bookkeeping/consistency validator is included:

```bash
python scripts/validate_mixed_ice_outputs.py mixed_ice_1p1d_outputs_example
```

The validator checks numerical and bookkeeping consistency, including finite/non-negative states, local CO partitioning, and cumulative release accounting.

## Parameter sweeps

Generate a sweep:

```bash
python scripts/mixed_ice_sweep.py generate \
    --base-yaml scripts/example_mist_params.yaml \
    --sweep-dir sweep_yamls \
    --run-root sweep_outputs \
    --model-script scripts/mixed_ice_transport_1p1d.py \
    --save-2d
```

Run it with:

```bash
bash sweep_yamls/commands.sh
```

and analyze it with:

```bash
python scripts/mixed_ice_sweep.py analyze \
    --sweep-dir sweep_yamls \
    --analysis-dir sweep_analysis
```

## Repository structure

```text
mist/
├── LICENSE
├── README.md
├── mist_logo.png
└── scripts/
    ├── mixed_ice_transport_1p1d.py
    ├── example_mist_params.yaml
    ├── analyse_run.py
    ├── mixed_ice_sweep.py
    └── validate_mixed_ice_outputs.py
```

## Scope and caveats

MIST is intended as a controlled model for studying matrix-dependent volatile survival and transport. It is **not** a complete thermo-chemical disk model.

In particular:
- radial transport is 1D
- the vertical structure is prescribed/reconstructed rather than dynamically evolved in 2D
- pebbles and small grains are represented by characteristic aerodynamic properties rather than a full grain-size distribution
- mixed-ice trapping and release are parameterized through effective reservoir fractions, release temperatures, transition widths, and host capacities
- no chemical reaction network is included
- laboratory trapping kinetics and microscopic ice morphology are not explicitly resolved

MIST is therefore best suited for testing qualitative transport behavior, model sensitivities, and physically motivated limits rather than making unique microscopic predictions for mixed-ice chemistry.

## Citation

A paper describing MIST and its application to matrix-dependent volatile transport in protoplanetary disks is currently in preparation.

If you use this repository before a formal citation is available, please cite the repository:

```text
https://github.com/j-p-terry/mist
```

This section will be updated with the paper citation and BibTeX entry when available.

## License

MIST is released under the **CC0 1.0 Universal** dedication. See `LICENSE` for details.
