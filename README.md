<!-- # The Generation Gap: What Using Generations Misses -->
<h1 align="center">The Generation Gap: What Using Generations Misses</h1>

<p style="text-align: center;">Ioannis Grigoriadis, Jacopo Michele di Matteo, Áron Richárd Ferencz, Lilly Schwarzenbach, Guszti Eiben · PPSN 2026
</p>


---

<!-- ## Abstract -->
<h2 align="center">Abstract</h2>

<p style="text-align: center;">Evolutionary computation has produced many successful al gorithms and tools. The main challenge in flexible evolutionary computation lies not only in varying  and selecting individuals, but also in how they are represented, stored, scheduled, and retrieved over time. This paper introduces ARIEL, a framework that shifts evolutionary computation from a generation focus to persistent, stateful individuals. We present three configurations: (1) synchronous, (2) archive-assisted, and (3) asynchronous. The experiments show that population management can support different evolutionary  workflows without changes to the underlying engine or operators. These workflows  can all be achieved within the same infrastructure by adjusting eligibility conditions and orchestra tion logic.</p>



---

## Method Overview

This repository implements **body-brain co-evolution** using a Neural Developmental Encoding (NDE) genotype to produce modular robot morphologies, paired with a CMA-ES-optimised ANN controller evaluated in [MuJoCo](https://mujoco.org/).

Three experimental variants are provided:

| Script | Description |
|---|---|
| `RE_sync.py` | Baseline — synchronous Body-Brain Evolution |
| `RE_async.py` | Asynchronous |
| `RE_JESUS.py` | Archive — detects stagnation and injects historically successful individuals from the archive. Comically named J.E.S.U.S. (Joint Evolutionary Strategies with Undead Sampling) during development |

All three share the same evaluation stack (`robot_worker.py`): a CMA-ES loop that optimises an ANN controller per robot and returns the minimum distance to target achieved.

---

## Repository Structure

```
PPSN_experiment/
├── RE_sync.py                  # Baseline experiment 
├── RE_async.py                 # Asynchronous experiment 
├── RE_JESUS.py                 # Archive experiment
├── robot_worker.py             # Shared MuJoCo evaluation worker
├── view_results_from_db.ipynb  # Results visualisation
├── ariel/                      # EA framework (git submodule)
└── pyproject.toml              # uv project config and dependency declarations
```

Output is written to `__data__/<experiment>/` as a SQLite database that can be explored with `view_results_from_db.ipynb`.

---

## Dependencies

- Python ≥ 3.12
- [MuJoCo](https://mujoco.org/) ≥ 3.3.6
- [ariel](ariel/) — included as a git submodule
- All other Python packages are listed in `pyproject.toml`

---

## Installation

```bash
# Clone with submodules
git clone --recurse-submodules <repo-url>
cd PPSN_experiment

# Install Python dependencies
uv sync
```

> **Note:** If you cloned without `--recurse-submodules`, run
> `git submodule update --init --recursive` before `uv sync`.

---

## Running

All three scripts share the same CLI. Run any of them with `--help` to see all options:

```bash
uv run python RE_sync.py --help
uv run python RE_async.py --help
uv run python RE_JESUS.py --help
```

### Common options

| Flag | Default | Description |
|---|---|---|
| `--pop-size` / `-p` | 50 | Population size |
| `--generations` / `-g` | 100 | Number of generations |
| `--modules` / `-m` | 20 | Max body modules |
| `--gene-size` | 64 | Genotype size per NDE layer |
| `--cma-gen` | 20 | CMA-ES generations per robot eval |
| `--cma-pop` | 10 | CMA-ES population size per robot eval |
| `--seed` / `-s` | *(random)* | Random seed for reproducibility |
| `--output` / `-o` | `__data__` | Output directory |

### Examples

```bash
# Run with defaults
uv run python RE_sync.py
uv run python RE_async.py
uv run python RE_JESUS.py

# Reproducible short run for testing
uv run python RE_async.py --generations 10 --pop-size 20 --seed 42

# Check the status of previous runs
uv run python RE_sync.py status
uv run python RE_async.py status
uv run python RE_JESUS.py status
```

Output is written to `__data__/<experiment>/` as a SQLite database.

---

## Visualising Results

Open `view_results_from_db.ipynb` in Jupyter and point it at the `.db` file produced in `__data__/`.

---

## Citation

Will be added once the paper is published

If you use this code please cite:

```bibtex
[INSERT CITATION HERE]

YES I LEFT THIS HERE UNTIL THE ACTUAL CITATION AS AS JOKE

HI
```
