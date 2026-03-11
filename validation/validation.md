# Validation Suite: Sato-Bers Cardiac Myocyte Model

## Overview

This validation suite verifies that the **Python** and **Julia** ports of the Sato-Bers ventricular cardiac myocyte model produce numerically identical results to the original **C++** implementation. All comparisons are performed at float64 (double) precision.

## Architecture

```
validation/
├── validation.md              # This document
├── run_all.sh                 # Master orchestration script
├── validate_export.cc         # C++ reference trace generator
├── validate_python.py         # Python cross-validation suite
├── validate_julia.jl          # Julia cross-validation suite
├── reference/                 # C++ reference CSVs (generated)
│   ├── scenario1_single_beat.csv
│   ├── scenario2_multi_beat.csv
│   ├── scenario3_apd_restitution.csv
│   ├── scenario4a_elevated_gk1.csv
│   ├── scenario4b_ical_block.csv
│   ├── scenario4c_enhanced_serca.csv
│   ├── scenario4d_depolarized.csv
│   ├── scenario5_stochastic.csv
│   └── scenario6_steady_state.csv
└── results/                   # Python/Julia traces (generated)
    ├── scenario1_python.csv
    ├── scenario5_python.csv
    └── ...
```

## How to Run

```bash
cd <project_root>
bash validation/run_all.sh
```

Or step by step:

```bash
# 1. Build and run C++ reference generator
g++ -std=c++23 -O2 -o validation/validate_export validation/validate_export.cc cell.cc
./validation/validate_export

# 2. Run Python validation
python3 validation/validate_python.py

# 3. Run Julia validation
julia validation/validate_julia.jl
```

## Validation Scenarios

### Scenario 1: Single-Beat Deterministic Trace

| Parameter | Value |
|-----------|-------|
| N_CaL | 0 (deterministic) |
| tauf | 30 ms |
| av | 3.0 |
| Pre-pacing | 100 beats |
| Recording | 1 beat (300 ms) |
| Sample rate | Every 10 steps (0.5 ms) |

**Purpose:** Verify equation-level correctness of all ion currents, calcium handling, and gating variable dynamics. This is the most fundamental test — if every equation is ported correctly, the state trajectories must match to machine precision.

**What it tests:**
- INa (Hodgkin-Huxley m³hj formulation)
- ICaL (Goldman-Hodgkin-Katz with d·f·q gating)
- IK1 (inward rectifier with Mg²⁺ block)
- IKr (rapid delayed rectifier with Rv inactivation)
- IKs (slow delayed rectifier)
- Ito (transient outward with Xto·Yto gating)
- IKp (plateau current)
- INCX (Na-Ca exchanger)
- SERCA uptake (Hill equation)
- SR release (piecewise Q function)
- Ca²⁺ buffering (Bsr, Bcd, troponin)
- Nernst potentials (ENa, EK, EKs)
- Forward Euler integration

**Expected tolerance:** < 1e-12 absolute and relative error.

### Scenario 2: Multi-Beat Convergence

| Parameter | Value |
|-----------|-------|
| N_CaL | 0 (deterministic) |
| tauf | 30 ms |
| Pre-pacing | 100 beats |
| Recording | 10 consecutive beats (3000 ms) |
| Sample rate | Every 20 steps (1.0 ms) |

**Purpose:** Verify that the model maintains a stable limit cycle without numerical drift accumulating across beat boundaries. Tests that the stimulus application logic (BCL timing, duration) is correctly implemented.

**What it tests:**
- Beat-to-beat stability
- No gradual drift in state variables
- Correct stimulus timing across multiple beats
- Calcium cycling periodicity

**Expected tolerance:** < 1e-12.

### Scenario 3: APD Restitution Curve

| Parameter | Value |
|-----------|-------|
| N_CaL | 0 (deterministic) |
| tauf | 20–60 ms (swept) |
| Pre-pacing | 200 beats per tauf |
| Measurement | 100 beats per tauf |
| Output | APD and peak [Ca²⁺]i per beat |

**Purpose:** Reproduce the main scientific output of the model — the relationship between ICaL inactivation kinetics (tauf) and action potential duration / calcium transient amplitude. This is the equivalent of **Figure 3** from Sato & Bers (2011) showing APD alternans as a function of tauf.

**What it tests:**
- APD measurement via linear interpolation at -70 mV threshold
- Peak Ca²⁺ tracking across beats
- Parameter sweep logic
- Steady-state convergence for each tauf value
- Transition between stable and alternating behavior

**Expected tolerance:** < 1e-8 (accumulated over many beats).

**Key physiology:**
- Low tauf (~20 ms): Fast ICaL recovery → short APD, large Ca transients
- High tauf (~60 ms): Slow ICaL recovery → long APD, stable
- Intermediate tauf: Alternans region where APD and Ca alternate beat-to-beat

### Scenario 4: Perturbation Tests

Four physiologically motivated parameter perturbations that stress-test different parts of the model.

#### 4a: Elevated IK1 Conductance (gk1 = 5.6, 2× baseline)

**Physiological motivation:** Simulates the effect of hyperkalemia (elevated extracellular K⁺). In vivo, hyperkalemia increases IK1 by shifting the reversal potential and increasing channel availability. We model this as doubled gk1 conductance.

**Expected effect:** Shortened APD, more negative resting potential, faster repolarization.

#### 4b: 50% ICaL Block (icabar = 1.75, 50% of 3.5)

**Physiological motivation:** Models the effect of L-type calcium channel blockers (e.g., nifedipine, verapamil). Clinically used as anti-arrhythmics and anti-hypertensives.

**Expected effect:** Shorter APD, significantly reduced Ca²⁺ transient, potentially altered alternans behavior.

#### 4c: Enhanced SERCA (vup = 0.500, 2× baseline)

**Physiological motivation:** Models the effect of SERCA overexpression or phospholamban knockout, both of which increase SR Ca²⁺ uptake rate. Relevant to heart failure research where SERCA is often downregulated.

**Expected effect:** Faster cytoplasmic Ca²⁺ decay, potentially larger SR load, altered SR release dynamics.

#### 4d: Depolarized Initial Condition (v₀ = -60 mV)

**Physiological motivation:** Tests model behavior starting from a non-resting state, as might occur during rapid pacing, ischemia, or in the context of partially depolarized tissue.

**Expected effect:** After 100 beats of pre-pacing, the model should converge to the same limit cycle as the baseline scenario. This tests the attracting nature of the limit cycle.

**All perturbation tolerances:** < 1e-11.

### Scenario 5: Stochastic Gating (Fixed Seed)

| Parameter | Value |
|-----------|-------|
| N_CaL | 100000 |
| tauf | 30 ms |
| RNG seed | 1821800813 |
| Pre-pacing | 50 beats |
| Recording | 1 beat |

**Purpose:** Verify that the stochastic gating implementation — specifically the **xorshift32 PRNG** and **Box-Muller transform** — produces identical random sequences across C++, Python, and Julia.

**What it tests:**
- xorshift32 bit manipulation (13/17/5 shift constants)
- Uniform-to-Gaussian conversion via Box-Muller
- Langevin noise amplitude calculation: σ = √(2·A·dt) where A = (α(1-x) + βx)/N
- Clamping of gate variables to [0, 1]
- Correct pairing of Box-Muller variates (za1→d, za2→f, fresh za1→q)

**Stochastic gating equations (Langevin):**
For gate variable x with forward rate α and backward rate β:
```
dx = (α(1-x) - βx)·dt + √(2·A·dt)·Z
where A = [α(1-x) + βx] / N_CaL
and Z ~ N(0,1) from Box-Muller transform
```

**Expected tolerance:** < 1e-10 (slightly relaxed due to accumulated float rounding in RNG).

### Scenario 6: Steady-State Fingerprint

| Parameter | Value |
|-----------|-------|
| N_CaL | 0 (deterministic) |
| tauf | 30 ms |
| Pre-pacing | 500 beats |
| Output | Single state vector |

**Purpose:** After very long pre-pacing, the model should be deeply converged to its limit cycle. The final state vector serves as a precise "fingerprint" of the attractor. Any transcription error in the model equations would cause this fingerprint to diverge.

**Expected tolerance:** < 1e-12.

## Tolerance Rationale

| Scenario | atol | rtol | Justification |
|----------|------|------|---------------|
| 1, 2, 6 | 1e-12 | 1e-12 | Deterministic, same equations → bit-level agreement |
| 3 | 1e-8 | 1e-8 | Accumulated over 200+100 beats × 41 tauf values |
| 4a–4d | 1e-11 | 1e-11 | Deterministic with parameter changes |
| 5 | 1e-10 | 1e-10 | Stochastic: RNG float casting can introduce ULP differences |

**Why float64?** The Sato-Bers model uses Forward Euler with dt=0.05 ms. While this is adequate for the physics, it means errors are O(dt²) per step and accumulate over millions of steps. Float64 provides ~15-16 significant digits, so after 6 million steps (1000 beats × 6000 steps/beat), we expect ~10-11 digits of agreement. Float32 would lose too much precision for meaningful cross-language comparison.

## Comparison Method

For each (time point, state variable) pair:

1. **Absolute error:** |ref - test|
2. **Relative error:** |ref - test| / max(|ref|, |test|, 1e-300)

A test passes if EITHER abs_err < atol OR rel_err < rtol. This dual-threshold approach handles both large and small values correctly:
- For V ≈ -90 mV: absolute error dominates
- For Xto ≈ 3.7e-5: relative error dominates

## Adding New Validation Scenarios

To add a new scenario:

1. Add the C++ scenario function in `validate_export.cc`
2. Call it from `main()` in validate_export.cc
3. Add the corresponding scenario function in `validate_python.py`
4. Add the corresponding scenario function in `validate_julia.jl`
5. Update this document

## Known Limitations

- **Stochastic mode tolerance:** The xorshift32 PRNG uses uint32 arithmetic which should be identical across languages, but the float64 conversion `(uint32 + 1.0) / (UINT32_MAX + 2.0)` may differ by 1 ULP depending on compiler/runtime. This is why scenario 5 uses relaxed tolerance.

- **Long runs:** The APD restitution scenario (3) is computationally expensive. On a laptop, expect ~5-10 minutes for the Python version and ~1 minute for Julia.

- **Pre-pacing adequacy:** 100–200 beats of pre-pacing may not be sufficient for convergence near the alternans bifurcation point. The validation uses 200 beats for the sweep, which is adequate for comparison purposes even if the absolute dynamics haven't fully converged.
