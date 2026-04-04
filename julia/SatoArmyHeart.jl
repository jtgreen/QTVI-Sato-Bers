"""
SatoArmyHeart.jl — ArmyHeart / Thunderbolt.jl compatible Sato-Bers ionic model.

Follows the same pattern as ArmyHeart's LuoRudy.jl:
  - mutable struct wrapping inner SatoBersModel + per-node RNG states
  - Adapt.@adapt_structure for CPU↔GPU transport (Vector{UInt32} → CuVector{UInt32})
  - cell_rhs! handles stimulus (from x, t) and Langevin ICaL gating noise (xorshift32)
  - NoStimulationProtocol() at MonodomainModel level (stimulus baked into cell_rhs!)
  - GPU: stochastic gating uses per-node RNG stored in rng_states[node_idx]

Usage:
    include("SatoArmyHeart.jl")
    ionic = SatoBersArmyModel(tauf=52.0, av=3.0, N_CaL=100000, ...)
    model = MonodomainModel(ConstantCoefficient(1.0), ConstantCoefficient(1.0),
                            ConstantCoefficient(SymmetricTensor{2,2,Float64}((0.1,0.0,0.1))),
                            NoStimulationProtocol(), ionic, :φₘ, :s)

DO NOT edit ArmyHeart.jl directly.
"""

include(joinpath(@__DIR__, "SatoBers.jl"))
using .SatoBers

using Thunderbolt
using Adapt
import Thunderbolt: AbstractIonicModel

# ============================================================================
# SatoBersArmyModel — Thunderbolt AbstractIonicModel wrapping SatoBers
# ============================================================================
"""
    SatoBersArmyModel{T, RV <: AbstractVector{UInt32}}

ArmyHeart/Thunderbolt-compatible ionic model for the Sato-Bers ventricular cell.

Fields:
  sato       — SatoBersModel (all biophysical parameters; immutable, stack-allocated)
  rng_states — per-node xorshift32 PRNG states (Vector on CPU, CuVector on GPU)
  nx, ny     — node grid dimensions (= n_elements + 1 in each direction)
  dx, dy     — node spacing (mm) for position → index mapping
  dt         — timestep (ms) for Langevin noise magnitude (set = DT₀ = 0.05 ms)
  bcl        — basic cycle length (ms) for periodic pacing
  stim_amp   — stimulus amplitude (µA/µF)
  stim_dur   — stimulus duration (ms)
  corner_mm  — pacing region: x < corner_mm AND y < corner_mm

GPU notes:
  Adapt.@adapt_structure converts rng_states: Vector{UInt32} → CuVector{UInt32}.
  Each CUDA thread i accesses rng_states[node_idx] exclusively (no contention).
"""
mutable struct SatoBersArmyModel{T, RV <: AbstractVector{UInt32}} <: AbstractIonicModel
    sato       :: SatoBersModel{T}   # inner model — all biophysics
    rng_states :: RV                 # per-node PRNG (one UInt32 per node)
    # Grid geometry for mapping position x → node index
    nx         :: Int
    ny         :: Int
    dx         :: T
    dy         :: T
    # Timestep for Langevin noise (must match outer integrator dt)
    dt         :: T
    # Pacing protocol (baked into cell_rhs! using x and t)
    bcl        :: T
    stim_amp   :: T
    stim_dur   :: T
    corner_mm  :: T
end

"""
    SatoBersArmyModel(; tauf, av, N_CaL, gna, gkr, gks, g_rel,
                        nx, ny, dx, dy, dt,
                        bcl, stim_amp, stim_dur, corner_mm)

Construct a SatoBersArmyModel with default biophysical parameters.
`nx × ny` must equal the number of FEM nodes (= n_elements + 1 per side).
"""
function SatoBersArmyModel(;
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
    dt        :: Float64 = 0.05,
    bcl       :: Float64 = 300.0,
    stim_amp  :: Float64 = 80.0,
    stim_dur  :: Float64 = 1.0,
    corner_mm :: Float64 = 1.0,
)
    sato = SatoBersModel{Float64}(
        tauf=tauf, av=av, N_CaL=N_CaL,
        gna=gna, gkr=gkr, gks=gks, g_rel=g_rel,
    )
    n = nx * ny
    seed = UInt32(0x6B8B4567)
    # UInt32 arithmetic wraps around mod 2^32 — correct for PRNG seeds
    rng_states = UInt32[seed ⊻ (UInt32(i % typemax(UInt32)) * UInt32(0x27220A95) + UInt32(0x374761A1)) for i in 1:n]
    return SatoBersArmyModel{Float64, Vector{UInt32}}(
        sato, rng_states, nx, ny, Float64(dx), Float64(dy), Float64(dt),
        Float64(bcl), Float64(stim_amp), Float64(stim_dur), Float64(corner_mm),
    )
end

# Adapt.jl: converts rng_states to CuVector on GPU
Adapt.@adapt_structure SatoBersArmyModel

# ============================================================================
# Thunderbolt AbstractIonicModel interface
# ============================================================================

Thunderbolt.transmembranepotential_index(::SatoBersArmyModel) = 1
Thunderbolt.num_states(::SatoBersArmyModel) = 15

function Thunderbolt.default_initial_state(m::SatoBersArmyModel, _=nothing)
    collect(SatoBers.default_initial_state(m.sato))
end

# ============================================================================
# GPU-safe inline xorshift32 (no heap allocation, no mutable struct)
# ============================================================================

@inline function xorshift32_step(x::UInt32)
    x ⊻= x << UInt32(13)
    x ⊻= x >> UInt32(17)
    x ⊻= x << UInt32(5)
    return x
end

@inline u32_to_uniform(x::UInt32) =
    (Float64(x) + 1.0) / (Float64(typemax(UInt32)) + 2.0)

# ============================================================================
# cell_rhs! — called per-node by Thunderbolt's ForwardEulerCellSolver
# ============================================================================

"""
    Thunderbolt.cell_rhs!(du, u, x, t, model::SatoBersArmyModel)

Compute ionic RHS + stochastic ICaL gate update for one FEM node.

  x  — 2D position vector (mm); used to determine stimulus region & node index
  t  — current simulation time (ms); used for periodic stimulus timing

Steps:
  1. Compute periodic corner stimulus from (x, t)
  2. Deterministic ionic RHS via cell_rhs_deterministic!
  3. Langevin noise on d, f, q gates (xorshift32 + Box-Muller)
     - Updates u[7,8,9] in-place; zeroes du[7,8,9]
     - ForwardEulerCellSolver then does u += dt*du (correct since du[7,8,9]=0)
"""
function Thunderbolt.cell_rhs!(
    du    :: AbstractVector,
    u     :: AbstractVector,
    x     :: AbstractVector,
    t     :: Real,
    model :: SatoBersArmyModel,
)
    # ---- 1. Periodic corner stimulus ----
    stim = zero(eltype(du))
    if x[1] < model.corner_mm && x[2] < model.corner_mm
        if mod(t, model.bcl) < model.stim_dur
            stim = model.stim_amp
        end
    end

    # ---- 2. Deterministic ionic RHS (all 15 state variables) ----
    cell_rhs_deterministic!(du, u, model.sato, stim)

    # ---- 3. Stochastic Langevin gating for ICaL gates (d=u[9], f=u[7], q=u[8]) ----
    N_CaL = model.sato.N_CaL
    if N_CaL > 0
        # Map 2D position → node index on regular Cartesian grid
        i_node = max(1, min(model.nx, round(Int, x[1] / model.dx) + 1))
        j_node = max(1, min(model.ny, round(Int, x[2] / model.dy) + 1))
        nidx   = (i_node - 1) * model.ny + j_node

        # Load per-node RNG state
        xsx = model.rng_states[nidx]

        # Extract current gate values and needed state
        v  = u[1]; cs = u[3]
        d  = u[9]; f  = u[7]; q  = u[8]
        tauf  = model.sato.tauf
        gam   = model.sato.gam
        dt    = model.dt
        N     = Float64(N_CaL)

        # -- Box-Muller pair 1: d gate (activation) and f gate (voltage inactivation) --
        xsx = xorshift32_step(xsx); r1 = u32_to_uniform(xsx)
        xsx = xorshift32_step(xsx); r2 = u32_to_uniform(xsx)
        bmag = sqrt(-2.0 * log(r1))
        za_d = bmag * cos(2π * r2)
        za_f = bmag * sin(2π * r2)

        # d gate update
        dinf = 1.0 / (1.0 + exp(-(v - 5.0) / 6.24))
        dp   = dinf / 5.0; dm = (1.0 - dinf) / 5.0
        Ad   = (dp*(1.0-d) + dm*d) / N
        u[9] = clamp(d + (dp*(1.0-d) - dm*d)*dt + sqrt(2.0*Ad*dt)*za_d, 0.0, 1.0)
        du[9] = 0.0

        # f gate update
        finf = 1.0 / (1.0 + exp((v + 35.0) / 8.6))
        fp   = finf / tauf; fm = (1.0 - finf) / tauf
        Af   = (fp*(1.0-f) + fm*f) / N
        u[7] = clamp(f + (fp*(1.0-f) - fm*f)*dt + sqrt(2.0*Af*dt)*za_f, 0.0, 1.0)
        du[7] = 0.0

        # -- Box-Muller pair 2: q gate (Ca²⁺-dependent inactivation) --
        xsx = xorshift32_step(xsx); r3 = u32_to_uniform(xsx)
        xsx = xorshift32_step(xsx); r4 = u32_to_uniform(xsx)
        bmag2 = sqrt(-2.0 * log(r3))
        za_q  = bmag2 * cos(2π * r4)

        qinf  = 1.0 / (1.0 + (cs/1.0)^gam)
        qp    = qinf / 20.0; qm = (1.0 - qinf) / 20.0
        Aq    = (qp*(1.0-q) + qm*q) / N
        u[8]  = clamp(q + (qp*(1.0-q) - qm*q)*dt + sqrt(2.0*Aq*dt)*za_q, 0.0, 1.0)
        du[8] = 0.0

        # Save updated RNG state (persists via pointer in CuVector on GPU)
        model.rng_states[nidx] = xsx
    end

    return nothing
end

println("SatoArmyHeart.jl loaded — SatoBersArmyModel ready (CPU + GPU via Adapt)")
