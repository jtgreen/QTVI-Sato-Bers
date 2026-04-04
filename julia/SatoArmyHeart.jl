"""
SatoArmyHeart.jl — ArmyHeart / Thunderbolt.jl compatible Sato-Bers ionic model.

Follows the same pattern as ArmyHeart's LuoRudy.jl / ToRORdEndo.jl:
  - mutable struct with CONSTANTS-like interface
  - Adapt.@adapt_structure for CPU↔GPU transport
  - cell_rhs! handles stimulus (via x, t) and Langevin gating noise
  - NoStimulationProtocol() used at MonodomainModel level
  - stochastic gating uses per-node xorshift32 RNG stored in rng_states vector

This file should be included/used from tissue_armyheart.jl.
DO NOT edit ArmyHeart.jl itself.

# GPU compatibility notes

rng_states is a plain Vector{UInt32} on CPU, CuVector{UInt32} on GPU.
Adapt.@adapt_structure converts between them.
CUDA kernel safety: each node i accesses rng_states[i] exclusively
(no contention), so mutations are thread-safe without atomics.
"""

include(joinpath(@__DIR__, "SatoBers.jl"))
using .SatoBers

using Thunderbolt
using Adapt
import Thunderbolt: AbstractIonicModel

# ============================================================================
# Parameters struct (mirrors ArmyHeart pattern with CONSTANTS-like access)
# ============================================================================

"""
    SatoBersParams{T}

All biophysical parameters of the Sato-Bers model, plus pacing protocol.
Stored as a flat NamedTuple for easy GPU adaptation.
"""
struct SatoBersParams{T}
    # Biophysical (mirrors SatoBersModel fields)
    F::T; F2::T; R::T; Temp::T
    K_in::T; Na_out::T; K_out::T; Ca_outmM::T; Na_in::T
    gna::T; gk1::T; gkr::T; gks::T; gto::T; gnaca::T; gkp::T; icabar::T
    tauf::T; av::T; gam::T; taujj::T
    cup::T; vup::T; g_rel::T
    # Pacing protocol
    bcl::T; stim_amp::T; stim_dur::T; corner_mm::T
    # Stochastic
    N_CaL::Int
    # Grid geometry (for RNG indexing)
    nx::Int; ny::Int; dx::T; dy::T
    # Timestep (needed for Langevin noise magnitude)
    dt::T
end

function SatoBersParams{T}(;
    tauf=T(52), av=T(3.0), N_CaL=100000,
    gna=T(12.0), gkr=T(0.0136), gks=T(0.0245), g_rel=T(75.0),
    bcl=T(300.0), stim_amp=T(80.0), stim_dur=T(1.0), corner_mm=T(1.0),
    nx=61, ny=61, dx=T(0.5), dy=T(0.5),
    dt=T(0.05),
) where T
    SatoBersParams{T}(
        T(96490), T(96.490), T(8.315), T(308),
        T(149.4), T(140), T(4), T(1.8), T(10),
        gna, T(2.8), gkr, gks, T(0.1), T(6.0), T(0.002216), T(3.5),
        tauf, av, T(0.7), T(1.0),
        T(0.5), T(0.250), g_rel,
        bcl, stim_amp, stim_dur, corner_mm,
        N_CaL, nx, ny, dx, dy, dt,
    )
end

# ============================================================================
# Main model struct — mutable for RNG state updates, Adapt-compatible
# ============================================================================

"""
    SatoBersArmyModel{T, RNGVec}

Thunderbolt.jl AbstractIonicModel for the Sato-Bers ventricular cell model.
Designed for ArmyHeart's 2D GPU-accelerated tissue simulation.

Fields:
  - `p`          : SatoBersParams (all biophysical + pacing parameters)
  - `rng_states` : per-node PRNG states (Vector{UInt32} on CPU, CuVector{UInt32} on GPU)
"""
mutable struct SatoBersArmyModel{T, RNGVec <: AbstractVector{UInt32}} <: AbstractIonicModel
    p          :: SatoBersParams{T}
    rng_states :: RNGVec
end

function SatoBersArmyModel(;
    tauf=52.0, av=3.0, N_CaL=100000,
    gna=12.0, gkr=0.0136, gks=0.0245, g_rel=75.0,
    bcl=300.0, stim_amp=80.0, stim_dur=1.0, corner_mm=1.0,
    nx=61, ny=61, dx=0.5, dy=0.5, dt=0.05,
)
    T   = Float64
    p   = SatoBersParams{T}(; tauf, av, N_CaL, gna, gkr, gks, g_rel,
                               bcl, stim_amp, stim_dur, corner_mm,
                               nx, ny, dx, dy, dt)
    n_nodes    = nx * ny
    base_seed  = UInt32(0x6B8B4567)
    rng_states = UInt32[base_seed ⊻ UInt32(i * 0x27220A95 + 0x374761A1) for i in 1:n_nodes]
    return SatoBersArmyModel{T, Vector{UInt32}}(p, rng_states)
end

Adapt.@adapt_structure SatoBersArmyModel

# ============================================================================
# Thunderbolt.jl AbstractIonicModel interface
# ============================================================================

Thunderbolt.transmembranepotential_index(::SatoBersArmyModel) = 1
Thunderbolt.num_states(::SatoBersArmyModel) = 15

function Thunderbolt.default_initial_state(m::SatoBersArmyModel, _=nothing)
    # Build a SatoBersModel from current parameters and get its default state
    sato = SatoBersModel{Float64}(
        tauf=m.p.tauf, av=m.p.av, N_CaL=0,
        gna=m.p.gna, gkr=m.p.gkr, gks=m.p.gks, g_rel=m.p.g_rel,
    )
    return collect(SatoBers.default_initial_state(sato))
end

# ============================================================================
# GPU-compatible xorshift32 (inline, no mutable structs)
# ============================================================================

@inline function xorshift32(x::UInt32)
    x ⊻= x << UInt32(13)
    x ⊻= x >> UInt32(17)
    x ⊻= x << UInt32(5)
    return x
end

@inline function u32_to_float(x::UInt32)
    # Map UInt32 → (0, 1) uniformly
    return (Float64(x) + 1.0) / (Float64(typemax(UInt32)) + 2.0)
end

# ============================================================================
# cell_rhs! — the core function called by Thunderbolt's ForwardEulerCellSolver
#
# Called per-node: cell_rhs!(du, u, x, t, model)
# ============================================================================

function Thunderbolt.cell_rhs!(
    du    :: AbstractVector,
    u     :: AbstractVector,
    x     :: AbstractVector,
    t     :: Real,
    model :: SatoBersArmyModel,
)
    p = model.p

    # ---- 1. Stimulus: applied to bottom-left corner (x < corner_mm, y < corner_mm) ----
    stim = zero(eltype(du))
    if x[1] < p.corner_mm && x[2] < p.corner_mm
        beat_phase = mod(t, p.bcl)
        if beat_phase < p.stim_dur
            stim = p.stim_amp
        end
    end

    # ---- 2. Deterministic ionic RHS ----
    # Unpack state (matches SatoBers state vector layout)
    v   = u[1];  ci  = u[2];  cs  = u[3];  cj  = u[4];  cjp = u[5]
    Ir  = u[6];  f   = u[7];  q   = u[8];  d   = u[9]
    h   = u[10]; jj  = u[11]; XKr = u[12]; XKs = u[13]; Xto = u[14]; Yto = u[15]

    # Build a temporary SatoBersModel for calling cell_rhs_deterministic!
    # (this is zero-allocation via inlining for the deterministic path)
    sato = SatoBersModel{Float64}(
        tauf  = p.tauf, av    = p.av,    N_CaL = 0,
        gna   = p.gna,  gkr   = p.gkr,  gks   = p.gks,
        gnaca = p.gnaca, icabar = p.icabar,
        g_rel = p.g_rel, gk1  = p.gk1,
        gto   = p.gto,  gkp   = p.gkp,
        cup   = p.cup,  vup   = p.vup,
        gam   = p.gam,  taujj = p.taujj,
        F     = p.F,    F2    = p.F2,   R     = p.R,   Temp   = p.Temp,
        K_in  = p.K_in, Na_out= p.Na_out, K_out = p.K_out,
        Ca_outmM = p.Ca_outmM, Na_in = p.Na_in,
    )
    cell_rhs_deterministic!(du, u, sato, stim)

    # ---- 3. Stochastic Langevin gating for ICaL d, f, q gates ----
    if p.N_CaL > 0
        # Map position x → node index in regular Cartesian grid
        i_node = max(1, min(p.nx, round(Int, x[1] / p.dx) + 1))
        j_node = max(1, min(p.ny, round(Int, x[2] / p.dy) + 1))
        node_idx = (i_node - 1) * p.ny + j_node

        # Load per-node RNG state
        xsx = model.rng_states[node_idx]

        # --- Box-Muller pair 1: d gate and f gate ---
        xsx = xorshift32(xsx); r1 = u32_to_float(xsx)
        xsx = xorshift32(xsx); r2 = u32_to_float(xsx)
        bmag = sqrt(-2.0 * log(r1))
        za1 = bmag * cos(2π * r2)   # for d gate
        za2 = bmag * sin(2π * r2)   # for f gate

        dt = p.dt
        N  = Float64(p.N_CaL)

        # d gate (activation)
        dinf   = 1.0 / (1.0 + exp(-(v - 5.0) / 6.24))
        taud   = 5.0
        dp     = dinf / taud
        dm     = (1.0 - dinf) / taud
        Ad     = (dp*(1.0-d) + dm*d) / N
        Δd     = (dp*(1.0-d) - dm*d)*dt + sqrt(2.0*Ad*dt)*za1
        u[9]   = clamp(d + Δd, 0.0, 1.0)
        du[9]  = 0.0   # stochastic update applied directly to u

        # f gate (voltage inactivation)
        tauf  = p.tauf
        finf  = 1.0 / (1.0 + exp((v + 35.0) / 8.6))
        fp    = finf / tauf
        fm    = (1.0 - finf) / tauf
        Af    = (fp*(1.0-f) + fm*f) / N
        Δf    = (fp*(1.0-f) - fm*f)*dt + sqrt(2.0*Af*dt)*za2
        u[7]  = clamp(f + Δf, 0.0, 1.0)
        du[7] = 0.0

        # --- Box-Muller pair 2: q gate ---
        xsx = xorshift32(xsx); r3 = u32_to_float(xsx)
        xsx = xorshift32(xsx); r4 = u32_to_float(xsx)
        bmag2 = sqrt(-2.0 * log(r3))
        za3   = bmag2 * cos(2π * r4)   # for q gate

        # q gate (Ca2+-dependent inactivation)
        cst    = 1.0; tauq = 20.0
        gam    = p.gam
        qinf   = 1.0 / (1.0 + (cs/cst)^gam)
        qp     = qinf / tauq
        qm     = (1.0 - qinf) / tauq
        Aq     = (qp*(1.0-q) + qm*q) / N
        Δq     = (qp*(1.0-q) - qm*q)*dt + sqrt(2.0*Aq*dt)*za3
        u[8]   = clamp(q + Δq, 0.0, 1.0)
        du[8]  = 0.0

        # Save updated RNG state
        model.rng_states[node_idx] = xsx
    end

    return nothing
end

println("SatoArmyHeart.jl loaded — SatoBersArmyModel ready (CPU + GPU via Adapt.jl)")
