"""
tissue_simulation.jl — Tissue-level simulation using ArmyHeart.jl

NOTE:
    This file requires the ArmyHeart.jl package (a submodule of this repository).
    ArmyHeart extends Thunderbolt.jl to support cardiac tissue simulations
    including cables, 2D tissue, pseudo-ECG computation, and QTVI analysis.

    ArmyHeart is not publicly available; it must be cloned via the submodule:
        git submodule update --init --remote

    Until ArmyHeart is available, this file documents the intended integration
    and the expected API usage.

# Overview

The tissue simulation couples the SatoBers single-cell model to a monodomain
PDE solver via the Thunderbolt.jl/ArmyHeart interface. This allows:

1. **Cable simulations** (1D): Study propagation velocity, APD gradients,
   and Ca2+ wave dynamics along a row of coupled myocytes.

2. **2D tissue simulations**: Model re-entrant arrhythmias, spiral waves,
   and the effect of electrical heterogeneity on wavefront stability.

3. **Pseudo-ECG computation**: ArmyHeart computes a far-field potential
   analogous to the clinical ECG, enabling direct comparison of simulated
   QTV with experimental/clinical measurements.

4. **QTVI analysis**: Beat-to-beat QT interval variability (QTVI) from the
   pseudo-ECG, as studied in Sato et al. (2025).

# Thunderbolt.jl Cell Model Interface

The SatoBers module already implements the interface expected by Thunderbolt.jl:

    # Required methods:
    num_states(model)              -> Int (= 15)
    default_initial_state(model)   -> SVector{15, T}
    transmembranepotential_index(::SatoBersModel) -> Int (= 1)
    cell_rhs!(du, u, model, stimulus)  -> Nothing

The cell_rhs! function computes derivatives for the coupled ODE/PDE system.
ArmyHeart uses this interface to integrate the cell model at each spatial node.

# Intended Usage (requires ArmyHeart)

```julia
# Activate the ArmyHeart environment
using ArmyHeart
include("SatoBers.jl")
using .SatoBers

# 1. Define the cell model
cell_model = SatoBersModel{Float64}(tauf=52.0, av=3.0, N_CaL=100000)

# 2. Set up a 1D cable (60 mm, Δx = 0.1 mm, 600 nodes)
cable = ArmyHeart.Cable1D(
    length = 60.0,          # mm
    dx     = 0.1,           # mm
    cell   = cell_model,
    D      = 0.001,         # cm²/ms (diffusion coefficient)
)

# 3. Run a cable simulation (10 beats, BCL = 300 ms)
results = ArmyHeart.simulate!(cable; bcl=300.0, beats=10, stim_site=1)

# 4. Compute pseudo-ECG from cable results
ecg = ArmyHeart.pseudo_ecg(results, electrode_position=30.0)

# 5. Compute QTVI from ECG
qtvi = ArmyHeart.qtvi(ecg)
println("QTVI = \$qtvi ms²")
```

# For 2D tissue:

```julia
# 2D tissue (60 × 60 mm²)
tissue = ArmyHeart.Tissue2D(
    size = (60.0, 60.0),
    dx   = 0.1,
    cell = cell_model,
    D    = 0.001,
)
results_2d = ArmyHeart.simulate!(tissue; bcl=300.0, beats=10)

# Pseudo-ECG and QTVI from 2D simulation
ecg_2d = ArmyHeart.pseudo_ecg_2d(results_2d, electrode=(30.0, 30.0, 3.0))
qtvi_2d = ArmyHeart.qtvi(ecg_2d)
```

# Key physics expected from tissue simulations

Based on the single-cell results in Sato et al. (2025):

1. **APD variability reduction in tissue**: Electrotonic coupling averages
   fluctuations, so tissue APD variability < single-cell APD variability.
   Heijman et al. (2013) predicted ~10-fold reduction.

2. **QTV near bifurcation**: Despite coupling, QTV on the pseudo-ECG still
   increases near the onset of alternans (both tauf and u bifurcations).

3. **Spatial patterns**: Near bifurcation, alternans can develop spatially
   discordant patterns (opposite-phase alternans in different regions),
   which creates substrate for re-entrant arrhythmias.

# References

- Sato D et al. "Beat-to-beat QT interval variability as a tool to detect
  the underlying cellular mechanisms of arrhythmias." J Physiol (2025).
  DOI: 10.1113/JP289051

- Heijman J et al. "Cellular and molecular electrophysiology of atrial
  fibrillation initiation, maintenance, and progression."
  Circ Res (2014).

- Thunderbolt.jl: https://github.com/termi-official/Thunderbolt.jl
"""

# --- Status message ---
@error """
tissue_simulation.jl requires ArmyHeart.jl (private submodule).

To enable tissue simulations:
  1. git submodule update --init --remote
  2. Uncomment the simulation code above

See the file docstring for complete usage instructions.
"""
