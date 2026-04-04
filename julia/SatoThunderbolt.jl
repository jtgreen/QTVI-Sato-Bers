"""
SatoThunderbolt.jl — Thunderbolt.jl-compatible wrapper for the Sato-Bers model.

Implements the `AbstractIonicModel` interface required by Thunderbolt.jl, which is
used by ArmyHeart.jl for 2D monodomain cardiac tissue simulation.

# How it fits into ArmyHeart / Thunderbolt

    using Thunderbolt
    using ArmyHeart
    include("SatoBers.jl");    using .SatoBers
    include("SatoThunderbolt.jl")

    ionic_model = SatoBersThunderboltModel(tauf=52.0, av=3.0, N_CaL=100000,
                                           nx=61, ny=61, dx=0.5, dy=0.5,
                                           bcl=300.0, stim_amp=80.0,
                                           stim_dur=1.0, corner_mm=1.0)

    monodomain = MonodomainModel(
        ConstantCoefficient(1.0),                                  # χ
        ConstantCoefficient(1.0),                                  # Cₘ
        ConstantCoefficient(SymmetricTensor{2,2,Float64}((0.1, 0.0, 0.1))),  # κ
        NoStimulationProtocol(),          # stimulus handled inside cell_rhs!
        ionic_model,
        :φₘ,
        :s,
    )

# Stimulus convention

Stimulus is baked into `cell_rhs!` using the position `x` and time `t` passed
by Thunderbolt's cell solver. This avoids the complexity of
`AnalyticalTransmembraneStimulationProtocol` with periodic beats.

# Stochastic gating

Per-cell Langevin noise is applied to the ICaL gates (d, f, q) inside
`cell_rhs!`. The per-cell PRNG state is stored in the model and indexed by
mapping position `x` to a grid node index. This assumes a regular Cartesian
grid with uniform node spacing.

# Thread safety

The per-cell RNG update (`rng_states[idx].xsx`) is done with a cell index
computed from position. If Thunderbolt uses multi-threading over DOFs, there
can be races on the RNG states. For safety, we rely on Thunderbolt's default
single-thread cell solve (or use the thread-local approach below).
Use `julia --threads=1` if you see non-deterministic results.
"""

include(joinpath(@__DIR__, "SatoBers.jl"))
using .SatoBers
using Thunderbolt

# ============================================================================
# SatoBersThunderboltModel — Thunderbolt-compatible ionic model
# ============================================================================

"""
    SatoBersThunderboltModel

Mutable struct wrapping `SatoBersModel` for use with Thunderbolt.jl's
`MonodomainModel`. Contains:
  - All biophysical parameters (via inner `SatoBersModel`)
  - Per-node PRNG states for stochastic ICaL gating
  - Pacing protocol parameters (BCL, stim amp, stim duration, corner region)
  - Grid geometry (for mapping position x → node index → RNG state)

# Constructor
    SatoBersThunderboltModel(;
        tauf=52.0, av=3.0, N_CaL=100000,
        gna=12.0, gkr=0.0136, gks=0.0245, g_rel=75.0,
        nx=61, ny=61, dx=0.5, dy=0.5,
        bcl=300.0, stim_amp=80.0, stim_dur=1.0, corner_mm=1.0
    )

where `nx×ny` is the FEM node grid (= number of elements + 1 in each direction),
`dx/dy` is the node spacing in mm, `corner_mm` is the size of the pacing region
(stimulus applied for x<corner_mm AND y<corner_mm).
"""
mutable struct SatoBersThunderboltModel{T} <: Thunderbolt.AbstractIonicModel
    # Inner Sato-Bers model (holds all biophysical parameters)
    sato   :: SatoBersModel{T}

    # Grid geometry for node indexing (regular Cartesian grid assumed)
    nx     :: Int     # Number of nodes in x direction (= n_elements_x + 1)
    ny     :: Int     # Number of nodes in y direction (= n_elements_y + 1)
    dx     :: T       # Node spacing in x (mm)
    dy     :: T       # Node spacing in y (mm)

    # Per-node PRNG states (one per grid node, indexed as (i-1)*ny + j)
    rng_states :: Vector{UInt32}

    # Pacing protocol
    bcl       :: T    # Basic cycle length (ms)
    stim_amp  :: T    # Stimulus amplitude (µA/µF)
    stim_dur  :: T    # Stimulus duration (ms)
    corner_mm :: T    # Pacing corner: stimulus applied when x<corner_mm AND y<corner_mm
end

function SatoBersThunderboltModel(;
    tauf      :: Float64 = 52.0,
    av        :: Float64 = 3.0,
    N_CaL     :: Int     = 100000,
    gna       :: Float64 = 12.0,
    gkr       :: Float64 = 0.0136,
    gks       :: Float64 = 0.0245,
    g_rel     :: Float64 = 75.0,
    nx        :: Int     = 61,
    ny        :: Int     = 61,
    dx        :: Float64 = 0.5,
    dy        :: Float64 = 0.5,
    bcl       :: Float64 = 300.0,
    stim_amp  :: Float64 = 80.0,
    stim_dur  :: Float64 = 1.0,
    corner_mm :: Float64 = 1.0,
)
    sato = SatoBersModel{Float64}(tauf=tauf, av=av, N_CaL=N_CaL,
                                   gna=gna, gkr=gkr, gks=gks, g_rel=g_rel)
    # Initialise per-node RNG seeds with unique values
    n_nodes    = nx * ny
    base_seed  = UInt32(0x6B8B4567)
    rng_states = [base_seed ⊻ UInt32(i * 0x27220A95 + 0x374761A1) for i in 1:n_nodes]
    return SatoBersThunderboltModel{Float64}(sato, nx, ny, dx, dy, rng_states,
                                             bcl, stim_amp, stim_dur, corner_mm)
end

# ============================================================================
# Thunderbolt.jl AbstractIonicModel interface
# ============================================================================

Thunderbolt.transmembranepotential_index(::SatoBersThunderboltModel) = 1
Thunderbolt.num_states(::SatoBersThunderboltModel) = 15

"""
    Thunderbolt.default_initial_state(model::SatoBersThunderboltModel, _)

Returns the default resting state vector for a single cell.
The second argument is unused (Thunderbolt may pass `dt` or `nothing`).
"""
function Thunderbolt.default_initial_state(model::SatoBersThunderboltModel, _)
    return collect(SatoBers.default_initial_state(model.sato))
end

# Thunderbolt also calls default_initial_state(model) without second arg in some paths
Thunderbolt.default_initial_state(model::SatoBersThunderboltModel) =
    Thunderbolt.default_initial_state(model, nothing)

# ============================================================================
# cell_rhs! — the core ionic RHS called per-node by Thunderbolt's cell solver
# ============================================================================

"""
    Thunderbolt.cell_rhs!(du, u, x, t, model::SatoBersThunderboltModel)

Compute the ionic RHS for one FEM node at position `x` and time `t`.

This function:
1. Computes the stimulus current from position and time.
2. Calls `cell_rhs_deterministic!` for the Sato-Bers ionic currents.
3. Applies Langevin noise to the ICaL gates (d, f, q) using the per-node PRNG.
4. Zeros out du[7], du[8], du[9] (already applied to u in-place).

Note: `u` is a *view* of the full solution vector for this node. Direct
mutation of `u[7,8,9]` for stochastic gate updates is intentional.
"""
function Thunderbolt.cell_rhs!(
    du :: AbstractVector,
    u  :: AbstractVector,
    x  :: AbstractVector,
    t  :: Real,
    model :: SatoBersThunderboltModel,
)
    # ---- 1. Stimulus current (corner pacing, periodic in time) ----
    stim = 0.0
    if x[1] < model.corner_mm && x[2] < model.corner_mm
        beat_phase = mod(t, model.bcl)
        if beat_phase < model.stim_dur
            stim = model.stim_amp
        end
    end

    # ---- 2. Deterministic ionic RHS ----
    cell_rhs_deterministic!(du, u, model.sato, stim)

    # ---- 3. Stochastic gating (Langevin noise on d, f, q gates) ----
    if model.sato.N_CaL > 0
        # Map position → node index (regular Cartesian grid)
        i = max(1, min(model.nx, round(Int, x[1] / model.dx) + 1))
        j = max(1, min(model.ny, round(Int, x[2] / model.dy) + 1))
        node_idx = (i - 1) * model.ny + j

        # Get per-node RNG state and advance it
        rng = StochasticState(model.rng_states[node_idx])

        # Apply Langevin noise: modifies u[7,8,9] in-place, zeroes du[7,8,9]
        # Note: Thunderbolt's ForwardEulerCellSolver will do u += dt*du after this,
        # so zeroing du[7,8,9] prevents double-application.
        stochastic_gate_update!(u, du, 0.05, model.sato, rng)   # dt=0.05 ms

        # Save updated RNG state
        model.rng_states[node_idx] = rng.xsx
    end

    return nothing
end

# ============================================================================
# Adapt.jl support (for GPU arrays)
# ============================================================================
# Note: Full GPU support requires moving rng_states to a CuArray and using
# atomic operations or per-thread RNGs. For CPU (single/multi-thread) this
# direct implementation is sufficient.

# If GPU support is needed later:
# using Adapt
# Adapt.@adapt_structure SatoBersThunderboltModel
# and change rng_states to a CuVector{UInt32}

println("SatoThunderbolt.jl loaded — SatoBersThunderboltModel ready for Thunderbolt.jl")
