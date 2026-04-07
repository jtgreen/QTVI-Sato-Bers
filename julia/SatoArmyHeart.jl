"""
SatoArmyHeart.jl — ArmyHeart / Thunderbolt.jl compatible Sato-Bers ionic model.

Architecture:
  - Immutable struct (required for CUDA kernel arguments)
  - rng_states can be Vector{UInt32} (CPU) or CuVector{UInt32} (GPU)
  - Explicit GPU constructor: SatoBersArmyModel(; ..., use_gpu=true)
  - cell_rhs! handles:
      * Periodic corner stimulus from (x, t)
      * Deterministic Sato-Bers ionic RHS
      * Stochastic Langevin ICaL gating (xorshift32 + Box-Muller) when N_CaL > 0
  - NoStimulationProtocol() at MonodomainModel level (stimulus baked into cell_rhs!)

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

Immutable (for GPU compatibility) ArmyHeart/Thunderbolt ionic model.

Fields:
  sato       — SatoBersModel{T} biophysical parameters (all bitstype)
  rng_states — per-node xorshift32 PRNG (Vector on CPU, CuVector on GPU)
  nx, ny     — node grid dimensions
  dx, dy     — node spacing (mm)
  dt         — timestep (ms) for Langevin noise scaling
  bcl        — basic cycle length (ms)
  stim_amp   — stimulus amplitude (µA/µF)
  stim_dur   — stimulus duration (ms)
  corner_mm  — pacing region: x < corner_mm AND y < corner_mm

Note: must be immutable struct so CUDA can pass it as kernel argument.
The rng_states array contents are still mutable (device memory writes work in CUDA).
"""
struct SatoBersArmyModel{T, RV <: AbstractVector{UInt32}} <: AbstractIonicModel
    sato       :: SatoBersModel{T}
    rng_states :: RV
    nx         :: Int
    ny         :: Int
    dx         :: T
    dy         :: T
    dt         :: T
    bcl        :: T
    stim_amp   :: T
    stim_dur   :: T
    corner_mm  :: T
end

"""
    SatoBersArmyModel(; tauf, av, N_CaL, ..., use_gpu=false)

Construct a SatoBersArmyModel. Set use_gpu=true to create with CuVector RNG states.
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
    use_gpu   :: Bool    = false,
)
    sato = SatoBersModel{Float64}(
        tauf=tauf, av=av, N_CaL=N_CaL,
        gna=gna, gkr=gkr, gks=gks, g_rel=g_rel,
    )
    n = nx * ny
    seed = UInt32(0x6B8B4567)
    rng_cpu = UInt32[seed ⊻ (UInt32(i % typemax(UInt32)) * UInt32(0x27220A95) + UInt32(0x374761A1)) for i in 1:n]

    if use_gpu
        error("use_gpu=true deprecated. Create CPU model then call Adapt.adapt(CuArray, model).")
    end
    rng_states = rng_cpu

    return SatoBersArmyModel{Float64, typeof(rng_states)}(
        sato, rng_states, nx, ny,
        Float64(dx), Float64(dy), Float64(dt),
        Float64(bcl), Float64(stim_amp), Float64(stim_dur), Float64(corner_mm),
    )
end

"""
    SatoBersArmyModel_gpu(ionic_cpu)

Create a GPU-ready version of an existing SatoBersArmyModel by moving rng_states to GPU.
Requires CUDA to be loaded.
"""
function SatoBersArmyModel_gpu(m::SatoBersArmyModel{T, Vector{UInt32}}) where T
    cu_rng = try
        CuVector{UInt32}(m.rng_states)
    catch e
        error("CUDA not available or CuVector not loaded. Load CUDA.jl first.\n$e")
    end
    return SatoBersArmyModel{T, typeof(cu_rng)}(
        m.sato, cu_rng, m.nx, m.ny,
        m.dx, m.dy, m.dt, m.bcl, m.stim_amp, m.stim_dur, m.corner_mm,
    )
end

# Adapt.jl: auto-adapt all AbstractArray fields (rng_states: Vector → CuArray on GPU)
# Use Adapt.@adapt_structure for correct recursive adaptation
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
# GPU-safe inline xorshift32
# ============================================================================

@inline function xorshift32_step(x::UInt32)
    x ⊻= x << UInt32(13)
    x ⊻= x >> UInt32(17)
    x ⊻= x << UInt32(5)
    return x
end

@inline function u32_to_uniform_f64(x::UInt32)
    (Float64(x) + 1.0) / (Float64(typemax(UInt32)) + 2.0)
end

# ============================================================================
# cell_rhs! — called per-node by Thunderbolt's ForwardEulerCellSolver
# ============================================================================

"""
    Thunderbolt.cell_rhs!(du, u, x, t, model::SatoBersArmyModel)

Compute ionic RHS + stochastic ICaL gate update for one FEM node.

  x  — 2D position (Vec{2,T} or AbstractArray); nothing → no position-dependent stim
  t  — current simulation time (ms)
"""
function Thunderbolt.cell_rhs!(
    du    :: AbstractVector,
    u     :: AbstractVector,
    x,
    t     :: Real,
    model :: SatoBersArmyModel,
)
    # ---- 1. Periodic corner stimulus ----
    stim = zero(eltype(du))
    if x !== nothing && x[1] < model.corner_mm && x[2] < model.corner_mm
        if mod(Float64(t), Float64(model.bcl)) < Float64(model.stim_dur)
            stim = eltype(du)(model.stim_amp)
        end
    end

    # ---- 2. Deterministic ionic RHS (all 15 state variables) ----
    cell_rhs_deterministic!(du, u, model.sato, Float64(stim))

    # ---- 3. Stochastic Langevin gating for ICaL gates (d=u[9], f=u[7], q=u[8]) ----
    N_CaL = model.sato.N_CaL
    if N_CaL > 0 && x !== nothing
        # Map 2D position → node index on regular Cartesian grid
        i_node = max(1, min(model.nx, round(Int, Float64(x[1]) / Float64(model.dx)) + 1))
        j_node = max(1, min(model.ny, round(Int, Float64(x[2]) / Float64(model.dy)) + 1))
        nidx   = (i_node - 1) * model.ny + j_node

        # Load per-node RNG state
        xsx = model.rng_states[nidx]

        # Extract current gate values
        v  = Float64(u[1]); cs = Float64(u[3])
        d  = Float64(u[9]); f  = Float64(u[7]); q  = Float64(u[8])
        tauf  = Float64(model.sato.tauf)
        gam   = Float64(model.sato.gam)
        dt    = Float64(model.dt)
        N     = Float64(N_CaL)

        # Box-Muller pair 1: d gate + f gate
        xsx = xorshift32_step(xsx); r1 = u32_to_uniform_f64(xsx)
        xsx = xorshift32_step(xsx); r2 = u32_to_uniform_f64(xsx)
        bmag = sqrt(-2.0 * log(r1))
        za_d = bmag * cos(2.0 * pi * r2)
        za_f = bmag * sin(2.0 * pi * r2)

        # d gate (activation)
        dinf = 1.0 / (1.0 + exp(-(v - 5.0) / 6.24))
        dp   = dinf / 5.0; dm = (1.0 - dinf) / 5.0
        Ad   = (dp*(1.0-d) + dm*d) / N
        d_new = clamp(d + (dp*(1.0-d) - dm*d)*dt + sqrt(2.0*Ad*dt)*za_d, 0.0, 1.0)
        u[9]  = eltype(du)(d_new)
        du[9] = zero(eltype(du))

        # f gate (voltage inactivation)
        finf = 1.0 / (1.0 + exp((v + 35.0) / 8.6))
        fp   = finf / tauf; fm = (1.0 - finf) / tauf
        Af   = (fp*(1.0-f) + fm*f) / N
        f_new = clamp(f + (fp*(1.0-f) - fm*f)*dt + sqrt(2.0*Af*dt)*za_f, 0.0, 1.0)
        u[7]  = eltype(du)(f_new)
        du[7] = zero(eltype(du))

        # Box-Muller pair 2: q gate (Ca-dependent inactivation)
        xsx = xorshift32_step(xsx); r3 = u32_to_uniform_f64(xsx)
        xsx = xorshift32_step(xsx); r4 = u32_to_uniform_f64(xsx)
        bmag2 = sqrt(-2.0 * log(r3))
        za_q  = bmag2 * cos(2.0 * pi * r4)

        qinf  = 1.0 / (1.0 + (cs/1.0)^gam)
        qp    = qinf / 20.0; qm = (1.0 - qinf) / 20.0
        Aq    = (qp*(1.0-q) + qm*q) / N
        q_new = clamp(q + (qp*(1.0-q) - qm*q)*dt + sqrt(2.0*Aq*dt)*za_q, 0.0, 1.0)
        u[8]  = eltype(du)(q_new)
        du[8] = zero(eltype(du))

        # Save updated RNG state (works for both CPU Vector and GPU CuVector)
        model.rng_states[nidx] = xsx
    end

    return nothing
end

println("SatoArmyHeart.jl loaded — SatoBersArmyModel (immutable, GPU-ready)")
