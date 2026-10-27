# Deadline Reliability Without Independence

A reproducible experiment package for deadline-constrained MDS-coded transmission over two binary Markov erasure paths. The evaluator calculates sharp upper and lower failure probabilities over causal couplings that preserve each path's conditional transition law. It also constructs an attaining coupling and searches for robust packet schedules.

## Quick start

Use Python 3.12 on a CPU system. The reference execution used ten worker processes on a system with 12 available logical CPUs and 18 GB RAM. The numerical kernels use float64 arithmetic. Worker count can be reduced to match available resources.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python tests/validate.py
python run_experiments.py --workers 10
python run_search_scaling.py --workers 10
python run_holdout.py --workers 10
python summarize.py
```

On Windows, activate the environment with `.venv\Scripts\activate`. Compilation is cached automatically by Numba. The first execution includes compilation time.

For an end-to-end rerun that compares outputs against the included reference results:

```bash
python verify_reproduction.py --workers 10
```

This command regenerates the numerical experiments in a temporary directory, checks probability values with explicit floating-point tolerances, checks simulation seeds and failure counts exactly, and writes `results/reproduction_check.json`. Execution times depend on the machine and system load.

## Files

- `src/model.py`: conditional marginal model, exact endpoint recursion, exhaustive schedule enumeration, attaining-coupling construction, vectorized forward simulation, and five-start exchange search.
- `config/experiment.json`: reference grid, asymmetric example, seeds, Monte Carlo sample sizes, and execution settings.
- `run_experiments.py`: complete 243-setting grid, all transmission budgets for 81 path pairs, ambiguity-radius sensitivity, 54 million Monte Carlo blocks, and evaluator timing.
- `run_search_scaling.py`: comparison of local search with every grid optimum, followed by nine larger-horizon searches.
- `run_holdout.py`: 200 additional random channel settings with exhaustive optimal schedules and the fixed local search.
- `tests/validate.py`: independent occupancy-flow linear programs, complete path enumeration, forward witness propagation, closed-form cases, monotonicity checks, and finite-field MDS rank checks.
- `summarize.py`: aggregate reductions, target-feasibility counts, Monte Carlo Wilson intervals, and timing summaries.
- `verify_reproduction.py`: automated end-to-end numerical comparison.
- `results/`: reference output tables, machine-readable summaries, validation reports, and runtime metadata.
- `CITATION.cff`: author and software citation metadata.
- `LICENSE`: MIT license.

## Model and notation

A path is successful in state 0 and erased in state 1. `marginal(p, burst)` returns the next-erasure probabilities conditional on the current state. The stationary erasure probability is `p`, and `burst` is the expected consecutive-erasure run length in slots. In the joint state `2*x+y`, the first bit is path A and the second is path B.

The conditional law of each path, given the entire joint past, equals its specified two-state transition law. Unknown dependence is represented by the probability `q` of both next states being erased. For marginal next-erasure probabilities `a,b`, the admissible interval is `[max(0,a+b-1), min(a,b)]`. The next-state probabilities in order `00,01,10,11` are `[1-a-b+q, b-q, a-q, q]`. The same interval construction applies to the initial stationary marginal probabilities. The joint initial distribution need not be stationary.

Schedules have one action per slot:

| Symbol | Integer | Action |
|---|---:|---|
| `.` | 0 | Idle |
| `A` | 1 | One symbol on A |
| `B` | 2 | One symbol on B |
| `X` | 3 | One distinct symbol on each path |

The entire source block is ready at slot zero. Any `k` received coded symbols recover it. Both paths evolve in idle slots, the sender uses a fixed schedule, and successful scheduled symbols arrive within their eligible slot.

The `risk` modes are `0` for independence, `1` for a fixed lower-endpoint coupling, `2` for a fixed upper-endpoint coupling, `3` for the sharp upper envelope, and `4` for the sharp lower envelope. The ambiguity radius interpolates the coupling interval between independence (`lam=0`) and the full admissible interval (`lam=1`). It is an uncertainty specification, not a correlation coefficient. The upper envelope permits a causal coupling to depend on time, channel state, and decoding progress. Its sharpness is with respect to that class.

## Experimental design

The main grid independently varies both paths' erasure probabilities over `{0.02,0.10,0.20}` and mean burst lengths over `{1.25,3,8}`, with `T=B=8` and `k` in `{2,4,6}`. All 12,870 schedules are evaluated in each of the 243 settings, giving 3,127,410 schedule instances. Objective ties are determined by tolerance `1e-12 * max(best_value, 1e-12)`. Nonrobust ties favor smaller robust risk, and robust ties favor smaller independent risk. The fixed search starts from all-A, all-B, alternating, front-loaded pairs, and uniformly spaced pairs.

The asymmetric illustration fixes `(pA,pB,LA,LB)=(0.1,0.2,4,2)`. At each source size, the three optimized schedules are simulated under independent, upper-endpoint, and attaining adverse couplings. Each configuration has 20 independently seeded batches of 100,000 blocks. Seeds are deterministically constructed from the base seed, block size, policy, coupling mode, and batch index in `run_experiments.py`.

The extra random instances use seed `240027`, uniform erasure probabilities on `[0.005,0.35]`, log-uniform burst lengths on `[1.05,12]`, and uniformly selected source sizes 2 through 7. These add 2,574,000 exhaustive schedule evaluations. Larger cases use `T` in `{16,32,64}`, `B=T`, `k=T/2` and three path configurations stated in `run_search_scaling.py`.

## Output guide

- `benchmark.csv`: six policies per grid setting, schedule patterns, independent and fixed-positive probabilities, and sharp upper/lower envelopes.
- `illustration.csv`: the same comparisons for the asymmetric example.
- `budget_frontiers.csv`: optimized schedules and probabilities for budgets 4 through 16 at source size 4; includes the 81 grid path pairs and the asymmetric illustration.
- `radius_sensitivity.csv`: exact minimax schedules at 41 uncertainty radii for each source size.
- `monte_carlo.csv`: every seed, trial count, failure count, and analytical probability.
- `monte_carlo_summary.csv`: aggregate estimates and Wilson 95% intervals for the 27 configurations.
- `search_timings.csv`, `evaluator_timings.csv`, `search_scaling.csv`: exhaustive evaluation, fixed-schedule evaluation, and local-search runtime measurements.
- `holdout.csv`: all 200 random test settings, exhaustive optima, local-search solutions, and optimality gaps.
- `validation.json`, `summary.json`, `run_metadata.json`: numerical validation, aggregate findings, and environment details.

Probability comparisons are deterministic up to floating-point rounding. With the pinned NumPy version, Monte Carlo failure counts are reproducible from the included seeds. Reported timing columns are machine-dependent. `optimum=-1` in larger-horizon search records means the result is a local-search solution; the associated `relative_gap` field is a neutral storage value, not a global-optimality claim.

## Main reference results

The median reduction in the worst-case failure envelope is 24.5843% across the main grid. The reduction is positive in 213 settings and zero in 30. The five-start search reaches the exact optimum in all 243 grid and 200 additional random settings. Of the 27 aggregated Monte Carlo Wilson 95% intervals, 26 contain the exact probability. The independent LP and path-enumeration comparisons agree within `1.3e-15`.

These results characterize the specified synthetic channel model and the causal-coupling uncertainty class. The nominal-performance cost, uncertainty-radius sensitivity, and feasibility tables are included to support explicit assessment of the tradeoffs.

## Authors and license

Lei Chen; Cemil Emre Yavas; Yue Zhang; Nancy R. Mead.

Correspondence: Lei Chen, `lchen@georgiasouthern.edu`.

Released under the MIT license. Citation metadata are provided in `CITATION.cff`.
