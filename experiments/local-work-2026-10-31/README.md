# Preserved local research

Archived on 31 October 2026 from a working tree based on commit
`f68b9b8998123cab521041bc05dc8e0d3d05b3cb`. The archive date does not establish
when each experiment was originally written.

| Location | Contents and purpose |
| --- | --- |
| `notebooks/XAI-Example-Income.ipynb` | The modified main Income example, including plotting and local explanation experiments. |
| `notebooks/XAI-Example-Income-2.ipynb` and `notebooks/XAI-Example-Cervical-2.ipynb` | RBO comparison and optimized LIME example variants. |
| `notebooks/XAI-Example-Income-Test.ipynb` and `notebooks/test22.ipynb` | Manual debugging variants with hard-coded examples and stray `kur` expressions. |
| `notebooks/Baseline RBO for XAI-Example-Income-Copy1.ipynb` | A baseline RBO variant using 25 in place of 64 for one calculation argument. |
| `notebooks/Example_Lime_Stability_in_Regression.ipynb` | A standalone regression example using CSI/VSI; this does not expand the application's supported scope. |
| `helpers/particle.py` | Particle representation for the swarm optimizer prototype. |
| `helpers/stability_utils.py` | Experimental LIME overrides and stability helpers; the application uses the installed `lime_stability` package. |
| `helpers/bayesian_optimization.py` | Original `util/utils.py`, containing Bayesian optimization helpers with an upstream attribution in the source. |
| `scripts/manual_demo.py` | Original `util/test.py`, a manual Income experiment, renamed to avoid discovery as an automated test. |
| `snippets/` | Original `stuff/GridSearch.py` and `stuff/PDPs.py`, preserved as fragments rather than supported standalone programs. |
| `requirements-copy.txt` | The historical dependency-file copy, preserved for context. |
| `patches/application-experiments.patch` | Original changes to `util/commons.py` and both dependency files, relative to the base commit. |
| `inventory.json` | Original paths, archive paths, transformations, and hashes. |

## What the optimizer experiment tried

The application patch compares the existing random/local kernel-width search
with particle swarm optimization, whose objective is the harmonic mean of CSI
(coefficient stability) and VSI (variable stability). It adds runtime logging
and starts an attempt to parallelize the original search. Fixed model settings,
two-fold cross-validation and seed 72 appear intended to support repeatable
experiments with fewer training runs.

## Unfinished work to resolve before promotion

- The preserved patch contains an empty `for process in n_jobs` loop and does
  not parse. Adding a loop body alone would still leave an integer being
  iterated and the original best-sample initialization removed.
- A multiprocessing design must account for the locally defined objective and
  its model state, especially with Windows process spawning.
- The swarm's result is logged, but the final LIME explanation uses the original
  optimizer's result. A comparison needs explicit selection and equal evaluation
  budgets; the two prototypes currently optimize different objective functions.
- `p.x_best = p.x` aliases mutable state. Parameter bounds and independent
  evaluation of the selected kernel also need attention.
- The patch forces SHAP's output index to zero and disables some dependency
  entries. These are preserved debugging/setup edits, not candidate defaults.
- Several notebooks contain undefined `kur` expressions, disabled cells and
  hard-coded instance indices. Clearing saved output does not repair these.

These files are review material. They have not been made runnable from their new
locations: notebook imports assume the project root is on Python's import path,
and the archived helpers are not integrated into the application. Do not apply
the complete historical patch as a fix. Select useful pieces into separate
changes after validating them.

The unchanged originals and saved results remain in the local backup described
in the [research index](../README.md).
