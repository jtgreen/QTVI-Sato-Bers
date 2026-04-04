"""
    SatoBers

Sato-Bers ventricular cardiac myocyte model — Julia implementation.

Ported from the C++ implementation (cell.h / cell.cc) and structured to be
compatible with Thunderbolt.jl's AbstractIonicModel interface pattern.

# Model overview

This module implements a single ventricular cardiac myocyte based on the
Sato-Bers model. It provides both deterministic and stochastic gating for
the L-type calcium channel (ICaL).

# Ion currents modeled

- **INa**  : Fast Na⁺ current (Hodgkin-Huxley formulation, m³hj gating)
- **ICaL** : L-type Ca²⁺ current (Goldman-Hodgkin-Katz flux equation, d·f·q gating)
- **IK1**  : Inward rectifier K⁺ current (instantaneous rectification)
- **IKr**  : Rapid delayed rectifier K⁺ current (XKr gating + rectification)
- **IKs**  : Slow delayed rectifier K⁺ current (XKs² gating)
- **Ito**  : Transient outward K⁺ current (Xto·Yto gating)
- **IKp**  : Plateau K⁺ current (instantaneous voltage dependence)
- **INCX** : Na⁺/Ca²⁺ exchanger current

# Calcium handling

- SERCA uptake (Hill equation, coefficient 2)
- SR Ca²⁺ release via piecewise-linear Q function (Sato et al.)
- Rapid-equilibrium Ca²⁺ buffering (SR sites, calmodulin, troponin)

# Stochastic gating

When `N_CaL > 0`, the ICaL gates (d, f, q) are updated using a Langevin
(channel noise) approach. Gaussian variates are generated via the Box-Muller
transform from uniform variates produced by a xorshift32 PRNG. The noise
magnitude scales as `1/√N_CaL`.

# Module structure

- `SatoBersModel{T}` — immutable struct holding all model parameters
- `cell_rhs_deterministic!` — computes the deterministic RHS (15 derivatives)
- `cell_rhs!` — alias for Thunderbolt.jl compatibility
- `stochastic_gate_update!` — applies Langevin noise to ICaL gates (d, f, q)
- `StochasticState` — mutable struct holding the xorshift32 PRNG state
- `xorshift!` — advance the xorshift32 PRNG

# State variables (15)

| Index | Symbol | Description                        | Units |
|-------|--------|------------------------------------|-------|
|  1    | v      | Membrane potential                 | mV    |
|  2    | ci     | Bulk cytosolic Ca²⁺                | µM    |
|  3    | cs     | Subspace (dyadic cleft) Ca²⁺       | µM    |
|  4    | cj     | Network SR Ca²⁺                    | µM    |
|  5    | cjp    | Junctional SR Ca²⁺                 | µM    |
|  6    | Ir     | SR Ca²⁺ release current            | µM/ms |
|  7    | f      | ICaL voltage inactivation gate     | —     |
|  8    | q      | ICaL Ca²⁺-dependent inactivation   | —     |
|  9    | d      | ICaL activation gate               | —     |
| 10    | h      | INa fast inactivation gate         | —     |
| 11    | j      | INa slow inactivation gate         | —     |
| 12    | XKr    | IKr activation gate                | —     |
| 13    | XKs    | IKs activation gate                | —     |
| 14    | Xto    | Ito activation gate                | —     |
| 15    | Yto    | Ito inactivation gate              | —     |

# References

Sato D, Bers DM. "How does stochastic ryanodine receptor-mediated
Ca leak fail to initiate a Ca spark?" Biophys J. 2011.

See also: `cell.h`, `cell.cc` — original C++ implementation.
"""
module SatoBers

using StaticArrays

export SatoBersModel, num_states, default_initial_state, transmembranepotential_index
export cell_rhs!, cell_rhs_deterministic!
export StochasticState, xorshift!, stochastic_gate_update!

# ============================================================================
# Model struct — compatible with Thunderbolt.jl AbstractIonicModel pattern
#
# This immutable struct holds all biophysical parameters for the Sato-Bers
# model. It is parameterised by type T (typically Float64) so that it can
# be used with automatic differentiation or other number types.
#
# Because the struct is immutable, changing a parameter (e.g., tauf) requires
# constructing a new SatoBersModel instance.
# ============================================================================
Base.@kwdef struct SatoBersModel{T}
    # Physical constants
    F::T   = T(96490)    # Faraday's constant (C/mol)
    F2::T  = T(96.490)   # Faraday's constant (C/mmol)
    R::T   = T(8.315)    # Gas constant (J/mol·K)
    Temp::T = T(308)     # Temperature (K)

    # Ionic concentrations (mM)
    K_in::T     = T(149.4)
    Na_out::T   = T(140)
    K_out::T    = T(4)
    Ca_outmM::T = T(1.8)
    Na_in::T    = T(10)

    # Maximum conductances / permeabilities for each ion current
    gna::T    = T(12.0)      # Max conductance for INa (fast Na⁺ current)
    gk1::T    = T(2.8)       # Max conductance for IK1 (inward rectifier K⁺)
    gkr::T    = T(0.0136)    # Max conductance for IKr (rapid delayed rectifier K⁺)
    gks::T    = T(0.0245)    # Max conductance for IKs (slow delayed rectifier K⁺)
    gto::T    = T(0.1)       # Max conductance for Ito (transient outward K⁺)
    gnaca::T  = T(6.0)       # Scaling factor for NCX (Na⁺/Ca²⁺ exchanger)
    gkp::T    = T(0.002216)  # Max conductance for IKp (plateau K⁺)
    icabar::T = T(3.5)       # Scaling factor for ICaL permeability

    # Model parameters
    tauf::T  = T(52)    # f-gate time constant (ms)
    av::T    = T(3.0)   # u (Ca²⁺ release parameter)
    gam::T   = T(0.7)   # gamma
    taujj::T = T(1.0)   # j-gate tau factor

    # SERCA
    cup::T = T(0.5)
    vup::T = T(0.250)   # 250 µM/s → 0.250 µM/ms

    # SR Ca²⁺ release gain (g_rel = 250 * 0.3 = 75.0 µM/ms by default)
    # Corresponds to GRyR in pharmacological intervention figures.
    # Increasing g_rel destabilizes Ca²⁺ cycling; decreasing it stabilizes.
    g_rel::T = T(75.0)

    # Stochastic gating
    N_CaL::Int = 100000
end

# ============================================================================
# Interface methods (Thunderbolt.jl pattern)
#
# These methods implement the minimal interface expected by Thunderbolt.jl
# for coupling an ionic model to a tissue-level PDE solver.
# ============================================================================

"""Return the index of the transmembrane potential in the state vector (1)."""
transmembranepotential_index(::SatoBersModel) = 1

"""Return the number of state variables in the model (15)."""
num_states(::SatoBersModel) = 15

"""
    default_initial_state(model::SatoBersModel{T}) -> SVector{15,T}

Return a static vector of 15 resting-state initial conditions matching
the C++ constructor defaults. Uses StaticArrays for stack allocation.
"""
function default_initial_state(::SatoBersModel{T}) where T
    @SVector T[
        -90.0,       # v
         0.5,        # ci
         0.5,        # cs
         115.0,      # cj
         115.0,      # cjp
         0.0,        # Ir
         0.1,        # f
         0.4,        # q
         0.001,      # d
         0.99869,    # h
         0.99887,    # j
         0.0,        # XKr
         0.0001,     # XKs
         3.742e-5,   # Xto
         1.0,        # Yto
    ]
end

# ============================================================================
# Core RHS — deterministic version
# ============================================================================
"""
    cell_rhs_deterministic!(du, u, model, stimulus)

Compute the deterministic RHS of the Sato-Bers model.
`u` is the state vector (length 15), `du` is filled with derivatives.
"""
function cell_rhs_deterministic!(du, u, model::SatoBersModel, stimulus=0.0)
    @inbounds begin
        # Unpack state variables from the state vector
        v   = u[1];  ci  = u[2];  cs  = u[3];  cj  = u[4];  cjp = u[5]
        Ir  = u[6];  f   = u[7];  q   = u[8];  d   = u[9]
        h   = u[10]; jg  = u[11]; XKr = u[12]; XKs = u[13]
        Xto = u[14]; Yto = u[15]

        # Unpack model parameters (destructured for concise notation)
        (; F, F2, R, Temp, K_in, Na_out, K_out, Ca_outmM, Na_in,
           gna, gk1, gkr, gks, gto, gnaca, gkp, icabar,
           tauf, av, gam, taujj, cup, vup, g_rel, N_CaL) = model

        # ===============================================================
        # Fast Sodium Current (INa) — Hodgkin-Huxley formulation
        # INa = gna * m³ * h * j * (V - E_Na)
        # m gate uses steady-state (minf) since activation is fast.
        # ===============================================================
        am   = 0.32 * (v + 47.13) / (1 - exp(-0.1 * (v + 47.13)))
        bm   = 0.08 * exp(-v / 11)
        minf = am / (am + bm)

        ah = 0.135 * exp((v + 80) / (-6.8))
        bh = 7.5 / (1 + exp(-0.1 * (v + 11)))
        dh = ah * (1 - h) - bh * h

        aj = 0.175 * exp((v + 100) / (-23)) / (1 + exp(0.15 * (v + 79)))
        bj = 0.3 / (1 + exp(-0.1 * (v + 32)))
        djg = (aj * (1 - jg) - bj * jg) / taujj

        # Nernst equilibrium potential for Na⁺
        ena = R * Temp / F2 * log(Na_out / Na_in)
        INa = gna * minf^3 * h * jg * (v - ena)

        # ===============================================================
        # Calcium Buffering — Rapid equilibrium approximation
        # beta = 1 / (1 + sum_i B_i * K_i / (Ca + K_i)²)
        # Three buffers: SR sites (Bsr/Ksr), calmodulin (Bcd/Kcd),
        # troponin (BT/KT).
        # ===============================================================
        Bsr = 47.0;  Ksr = 0.6
        Bcd = 24.0;  Kcd = 7.0
        k_on = 0.0377;  k_off = 0.0196
        BT = 70.0;  KT = k_off / k_on
        bci = 1 / (1 + Bsr*Ksr/(ci+Ksr)^2 + Bcd*Kcd/(ci+Kcd)^2 + BT*KT/(ci+KT)^2)
        bcs = 1 / (1 + Bsr*Ksr/(cs+Ksr)^2 + Bcd*Kcd/(cs+Kcd)^2 + BT*KT/(cs+KT)^2)

        # ===============================================================
        # L-type Calcium Current (ICaL) — Goldman-Hodgkin-Katz flux eq.
        # Uses GHK because Ca²⁺ concentrations differ by orders of
        # magnitude across the membrane. Gated by d·f·q.
        # ===============================================================
        Pca = 2.7/3.5 * 0.00000054  # Ca²⁺ permeability (cm/s)
        vF_RT = (v/1000) * F / R / Temp
        jca = d * f * q * icabar * Pca *
              (4 * (v/1000) * F * F / R / Temp) *
              ((cs/1000) * exp(2*vF_RT) - 0.341*Ca_outmM) /
              (exp(2*vF_RT) - 1)

        # ===============================================================
        # ICaL Gating (deterministic relaxation: dx/dt = (x∞ - x)/τ)
        # ===============================================================
        dinf = 1 / (1 + exp(-(v - 5.0) / 6.24))
        taud = 5.0
        finf = 1 / (1 + exp((v + 35.0) / 8.6))
        cst  = 1.0
        qinf = 1 / (1 + (cs/cst)^gam)
        tauq = 20.0

        dd = (dinf - d) / taud
        df = (finf - f) / tauf
        dq = (qinf - q) / tauq

        # ===============================================================
        # Sodium-Calcium Exchanger (NCX) — 3Na⁺:1Ca²⁺ exchange
        # Voltage-dependent with partition coefficient xi = 0.35.
        # ===============================================================
        Kmna = 87.5;  Kmca = 1.380;  xi = 0.35;  ksat = 0.1
        a = (v/1000) * F / R / Temp
        jnaca = gnaca / (Kmna^3 + Na_out^3) / (Kmca + Ca_outmM) *
                (exp(xi*a) * Na_in^3 * Ca_outmM -
                 exp((xi-1)*a) * Na_out^3 * (cs/1000)) /
                (1 + ksat * exp((xi-1)*a))

        # ===============================================================
        # SERCA Uptake — Hill equation (coefficient 2)
        # ===============================================================
        jup = vup * ci^2 / (ci^2 + cup^2)

        # ===============================================================
        # SR Ca²⁺ Release — Piecewise-linear Q function (Sato et al.)
        # Q depends on jSR load (cjp): Q=0 below 50, linear ramp 50-110,
        # steeper slope (av) above 110. w ensures continuity at cjp=110.
        # ===============================================================
        w = 1.5 * (110 - 50) - av * 110
        Q = if cjp < 50
                0.0
            elseif cjp >= 110
                av * cjp + w
            else
                1.5 * (cjp - 50)
            end
        Q *= 0.001

        taur   = 20.0        # Release current decay time constant (ms)
        dIr = -g_rel * jca * Q / icabar - Ir / taur  # Trigger + exponential decay

        # ===============================================================
        # Inward Rectifier Potassium Current (IK1)
        # Stabilises resting potential near E_K with inward rectification.
        # ===============================================================
        EK   = R * Temp / F2 * log(K_out / K_in)
        KmK1 = 13.0
        k1inf = 1 / (2 + exp(1.62 * F2 / R / Temp * (v - EK)))
        IK1   = gk1 * k1inf * K_out / (K_out + KmK1) * (v - EK)

        # ===============================================================
        # Rapid Delayed Rectifier K⁺ Current (IKr) — phase 3 repolarisation
        # ===============================================================
        Rv  = 1 / (1 + 2.5 * exp(0.1 * (v + 28)))
        IKr = gkr * Rv * XKr * sqrt(K_out / 4) * (v - EK)
        XKrinf = 1 / (1 + exp(-2.182 - 0.1819*v))
        tauKr  = 43 + 1 / (exp(-5.495 + 0.1691*v) + exp(-7.677 - 0.0128*v))
        dXKr   = (XKrinf - XKr) / tauKr

        # ===============================================================
        # Slow Delayed Rectifier K⁺ Current (IKs)
        # Mixed K⁺/Na⁺ reversal (P_Na/P_K = 0.01833), XKs² gating.
        # ===============================================================
        EKs    = R * Temp / F2 * log((K_out + 0.01833*Na_out) / (K_in + 0.01833*Na_in))
        IKs    = gks * XKs^2 * (v - EKs)
        XKsinf = 1 / (1 + exp((v - 16) / (-13.6)))
        tauKs  = 1 / ((0.0000719*(v-10)) / (1 - exp(-0.148*(v-10))) +
                       (0.000131*(v-10)) / (exp(0.0687*(v-10)) - 1))
        dXKs   = (XKsinf - XKs) / tauKs

        # ===============================================================
        # Transient Outward K⁺ Current (Ito) — phase 1 notch
        # Xto = activation, Yto = inactivation (HH alpha/beta rates).
        # ===============================================================
        Ito  = gto * Xto * Yto * (v - EK)
        aXto = 0.04516 * exp(0.03577*v)
        bXto = 0.0989 * exp(-0.06237*v)
        dXto = aXto * (1 - Xto) - bXto * Xto
        aYto = 0.005415 * exp((v+33.5)/(-5)) / (1 + 0.051335*exp((v+33.5)/(-5)))
        bYto = 0.005415 * exp((v+33.5)/5)    / (1 + 0.051335*exp((v+33.5)/5))
        dYto = aYto * (1 - Yto) - bYto * Yto

        # ===============================================================
        # Plateau K⁺ Current (IKp) — small, time-independent
        # ===============================================================
        KKp = 1 / (1 + exp((7.488 - v) / 5.98))
        IKp = gkp * KKp * (v - EK)

        # ===============================================================
        # Calcium Concentration ODEs
        # ci=cytosol, cs=subspace, cjp=jSR, cj=NSR
        # vivs = volume ratio (cytosol/subspace), taus = diffusion τ (ms)
        # ===============================================================
        vivs = 10.0;  taus = 2.0
        dci  = bci * ((cs - ci)/taus - jup)
        dcs  = bcs * (vivs * (Ir - (cs - ci)/taus - jca + jnaca))
        taua = 50.0
        dcjp = (cj - cjp) / taua
        dcj  = -Ir + jup

        # ===============================================================
        # Membrane Potential ODE: dV/dt = -(I_ion - I_stim)
        # Ca²⁺ fluxes converted to current: 0.02 * 1000 factor (µM→mM).
        # Factor 2 on jca for the 2+ valence of Ca²⁺.
        # ===============================================================
        dv = -(-stimulus + INa + IK1 + IKr + IKs + Ito + IKp + 0.02*(jnaca + 2*jca)*1000)

        # --- Fill derivative vector (same ordering as state vector) ---
        du[1]  = dv;    du[2]  = dci;   du[3]  = dcs
        du[4]  = dcj;   du[5]  = dcjp;  du[6]  = dIr
        du[7]  = df;    du[8]  = dq;    du[9]  = dd
        du[10] = dh;    du[11] = djg;   du[12] = dXKr
        du[13] = dXKs;  du[14] = dXto;  du[15] = dYto
    end
    return nothing
end

# Alias for Thunderbolt.jl compatibility — the default cell_rhs! uses the
# deterministic formulation. For stochastic gating, call
# stochastic_gate_update! separately after computing du.
cell_rhs!(du, u, model::SatoBersModel, stimulus=0.0) =
    cell_rhs_deterministic!(du, u, model, stimulus)

# ============================================================================
# Stochastic gating support (standalone use)
#
# These types and functions implement Langevin (channel noise) updates for
# the ICaL gates (d, f, q). They are used in standalone simulations;
# Thunderbolt.jl tissue simulations typically use deterministic gating only.
# ============================================================================

"""Mutable state for the xorshift32 pseudo-random number generator."""
mutable struct StochasticState
    xsx::UInt32
end

"""
    xorshift!(s::StochasticState) -> UInt32

Advance the xorshift32 PRNG (Marsaglia, 2003) by one step and return the
new state. Shift constants (13, 17, 5) match the C++ implementation exactly,
ensuring reproducible sequences across languages from the same seed.
"""
function xorshift!(s::StochasticState)
    x = s.xsx
    x ⊻= x << 13
    x ⊻= x >> 17
    x ⊻= x << 5
    s.xsx = x
    return x
end

"""
    stochastic_gate_update!(u, du, dt, model, rng_state)

After computing `du` via `cell_rhs_deterministic!`, call this to apply
Langevin noise to the d, f, q gates (indices 9, 7, 8).
Modifies `u` in place for those gates; the caller should then do
`u .+= du .* dt` for all other variables.
"""
function stochastic_gate_update!(u, du, dt, model::SatoBersModel, rng::StochasticState)
    N_CaL = model.N_CaL
    N_CaL == 0 && return nothing   # Skip if deterministic mode

    v  = u[1];  cs = u[3]
    d  = u[9];  f  = u[7];  q  = u[8]

    tauf = model.tauf
    gam  = model.gam

    # ---------------------------------------------------------------
    # Box-Muller transform: generate pairs of N(0,1) Gaussian variates
    # from pairs of Uniform(0,1) variates produced by xorshift32.
    #   Z1 = sqrt(-2 ln U1) * cos(2π U2)
    #   Z2 = sqrt(-2 ln U1) * sin(2π U2)
    # ---------------------------------------------------------------
    # Box-Muller pair 1 (za1 for d gate, za2 for f gate)
    r1 = (Float64(xorshift!(rng)) + 1.0) / (Float64(typemax(UInt32)) + 2.0)
    r2 = (Float64(xorshift!(rng)) + 1.0) / (Float64(typemax(UInt32)) + 2.0)
    mag = sqrt(-2.0 * log(r1))
    za1 = mag * cos(2π * r2)
    za2 = mag * sin(2π * r2)

    # --- d gate (activation) — Langevin update ---
    # alpha = x_inf/tau (opening rate), beta = (1-x_inf)/tau (closing rate)
    # A = (alpha*(1-x) + beta*x) / N_CaL  (diffusion coefficient)
    # dx = drift*dt + sqrt(2*A*dt) * Z  (Langevin step)
    dinf    = 1 / (1 + exp(-(v - 5.0) / 6.24))
    taud    = 5.0
    d_plus  = dinf / taud       # Opening rate
    d_minus = (1 - dinf) / taud # Closing rate
    Ad      = (d_plus*(1-d) + d_minus*d) / N_CaL  # Diffusion coefficient
    Δd      = (d_plus*(1-d) - d_minus*d)*dt + sqrt(2*Ad*dt)*za1
    u[9]    = clamp(d + Δd, 0.0, 1.0)  # Clamp gate to [0, 1]

    # --- f gate (voltage inactivation) — Langevin update ---
    finf    = 1 / (1 + exp((v + 35.0) / 8.6))
    f_plus  = finf / tauf
    f_minus = (1 - finf) / tauf
    Af      = (f_plus*(1-f) + f_minus*f) / N_CaL
    Δf      = (f_plus*(1-f) - f_minus*f)*dt + sqrt(2*Af*dt)*za2
    u[7]    = clamp(f + Δf, 0.0, 1.0)

    # Box-Muller pair 2 (only za1 needed for the q gate)
    r1 = (Float64(xorshift!(rng)) + 1.0) / (Float64(typemax(UInt32)) + 2.0)
    r2 = (Float64(xorshift!(rng)) + 1.0) / (Float64(typemax(UInt32)) + 2.0)
    mag = sqrt(-2.0 * log(r1))
    za1 = mag * cos(2π * r2)

    # --- q gate (Ca²⁺-dependent inactivation) — Langevin update ---
    cst     = 1.0     # Reference [Ca²⁺] for q-gate (µM)
    tauq    = 20.0    # q-gate time constant (ms)
    qinf    = 1 / (1 + (cs/cst)^gam)
    q_plus  = qinf / tauq
    q_minus = (1 - qinf) / tauq
    Aq      = (q_plus*(1-q) + q_minus*q) / N_CaL
    Δq      = (q_plus*(1-q) - q_minus*q)*dt + sqrt(2*Aq*dt)*za1
    u[8]    = clamp(q + Δq, 0.0, 1.0)

    # Zero out the deterministic derivatives for d, f, q since we already
    # applied the stochastic update directly to u. The caller will do
    # u .+= du .* dt for all other state variables.
    du[7] = 0.0
    du[8] = 0.0
    du[9] = 0.0

    return nothing
end

end # module
