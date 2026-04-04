#!/usr/bin/env julia
"""
generate_pharm_data.jl — Generate pharmacological intervention figure data
for Sato et al. (2025) Figs 5–10 and Appendix Fig A3.

Reproduces the key finding that:
- Interventions targeting Ca²⁺ cycling (GRyR) selectively reduce APD variability
  in Ca²⁺-driven instability scenarios
- Interventions targeting membrane voltage (GKr, GKs) selectively reduce APD
  variability in voltage-driven instability scenarios
- GCaL affects both instability types
- GNa has minimal effect on either

Produces CSV files in figures/data/:
  - pharm_grel_sweep.csv   : APD variability vs GRyR scaling (Figs A3A / 5)
  - pharm_gkr_sweep.csv    : APD variability vs GKr scaling  (Figs A3B / 6)
  - pharm_gks_sweep.csv    : APD variability vs GKs scaling  (Figs A3C / 7)
  - pharm_gcal_sweep.csv   : APD variability vs GCaL scaling (Fig 8)
  - pharm_gna_sweep.csv    : APD variability vs GNa scaling  (Fig 9)

Two baseline conditions are compared:
  (a) Voltage-driven instability: tauf = 50 ms, u = 3.0  (near voltage alternans)
  (b) Ca²⁺-driven instability:   tauf = 35 ms, u = 9.0  (near Ca²⁺ alternans)
"""

include(joinpath(@__DIR__, "..", "julia", "SatoBers.jl"))
using .SatoBers

const dt   = 0.05
const BCL  = 300.0
const BCLn = round(Int, BCL / dt)
const Durn = round(Int, 1.0 / dt)
const STIM = 50.0
const VC   = -70.0

# Number of beats for pre-pacing and measurement
const PREPACE_BEATS = 300
const MEAS_BEATS    = 500

"""pace! — advance state one time step with optional stimulus."""
function pace!(u, du, model, rng, st)
    cell_rhs_deterministic!(du, u, model, st)
    stochastic_gate_update!(u, du, dt, model, rng)
    @. u += du * dt
end

"""Prepace for `beats` beats and return state."""
function prepace!(u, du, model, rng, beats)
    for _ in 1:beats
        for tn in 0:BCLn-1
            pace!(u, du, model, rng, tn < Durn ? STIM : 0.0)
        end
    end
end

"""
    measure_beats!(u, du, model, rng, beats) -> (APDs, CaMaxes)

Run `beats` beats, recording APD (at V = -70 mV threshold) and peak Ca per beat.
"""
function measure_beats!(u, du, model, rng, beats)
    APDs  = Float64[]
    CaMax = Float64[]

    first   = false
    cimax   = 0.0
    apd     = 0.0
    APDt1   = 0.0
    APDt2   = 0.0
    vold    = u[1]

    for tn in 0:(beats * BCLn - 1)
        t = tn * dt
        if mod(tn, BCLn) < Durn
            if first
                first = false
                push!(APDs,  apd)
                push!(CaMax, cimax)
                cimax = 0.0
                apd   = 0.0
            end
            pace!(u, du, model, rng, STIM)
        else
            first = true
            pace!(u, du, model, rng, 0.0)
        end
        if u[2] > cimax
            cimax = u[2]
        end
        if vold < VC && u[1] > VC
            APDt1 = (t - dt) + dt * (VC - vold) / (u[1] - vold)
        elseif vold > VC && u[1] < VC
            APDt2 = (t - dt) + dt * (VC - vold) / (u[1] - vold)
            apd   = APDt2 - APDt1
        end
        vold = u[1]
    end
    return APDs, CaMax
end

"""
APD variability statistics. Returns (mean, std) of APD.
"""
function apd_stats(apds)
    isempty(apds) && return (NaN, NaN)
    mu = sum(apds) / length(apds)
    var = sum((x - mu)^2 for x in apds) / max(length(apds) - 1, 1)
    return (mu, sqrt(var))
end

"""
    run_condition(model; prepace=PREPACE_BEATS, meas=MEAS_BEATS) -> (apd_mean, apd_std)

Run a single simulation condition and return APD statistics.
"""
function run_condition(model; prepace=PREPACE_BEATS, meas=MEAS_BEATS)
    u_ref = collect(default_initial_state(model))
    u  = copy(u_ref)
    du = zeros(15)
    rng = StochasticState(UInt32(1821800813))
    prepace!(u, du, model, rng, prepace)
    APDs, _ = measure_beats!(u, du, model, rng, meas)
    return apd_stats(APDs)
end

# ============================================================================
# Baseline conditions
# ============================================================================

# (a) Voltage-driven instability: tauf = 50 ms, u = 3.0
const TAUF_V = 50.0   # High τ_f → near voltage alternans onset
const U_V    = 3.0

# (b) Ca²⁺-driven instability: tauf = 35 ms, u = 9.0
const TAUF_CA = 35.0
const U_CA    = 9.0   # High u → near Ca²⁺ alternans onset

mkpath(joinpath(@__DIR__, "data"))

println("=" ^ 70)
println("Pharmacological intervention sweeps")
println("Voltage-driven baseline: tauf=$(TAUF_V) ms, u=$(U_V)")
println("Ca²⁺-driven baseline:   tauf=$(TAUF_CA) ms, u=$(U_CA)")
println("Pre-pace: $(PREPACE_BEATS) beats, Measurement: $(MEAS_BEATS) beats")
println("=" ^ 70)

# ============================================================================
# Baseline APD variability (control)
# ============================================================================
println("\n--- Computing baselines ---")

model_v_ctrl  = SatoBersModel{Float64}(tauf=TAUF_V,  av=U_V,   N_CaL=100000)
model_ca_ctrl = SatoBersModel{Float64}(tauf=TAUF_CA, av=U_CA,  N_CaL=100000)

mu_v_ctrl,  sig_v_ctrl  = run_condition(model_v_ctrl,  prepace=400)
mu_ca_ctrl, sig_ca_ctrl = run_condition(model_ca_ctrl, prepace=400)

println("  Voltage-driven baseline: APD = $(round(mu_v_ctrl,  digits=2)) ± $(round(sig_v_ctrl,  digits=3)) ms")
println("  Ca²⁺-driven baseline:   APD = $(round(mu_ca_ctrl, digits=2)) ± $(round(sig_ca_ctrl, digits=3)) ms")

# ============================================================================
# 1. GRyR sweep (g_rel scaling) — Fig A3A / Fig 5
# ============================================================================
println("\n=== GRyR sweep (Fig A3A / 5) ===")
open(joinpath(@__DIR__, "data", "pharm_grel_sweep.csv"), "w") do io
    println(io, "grel_scale,apd_mean_v,apd_std_v,apd_mean_ca,apd_std_ca")
    # Add baseline row
    println(io, "1.0,$(mu_v_ctrl),$(sig_v_ctrl),$(mu_ca_ctrl),$(sig_ca_ctrl)")

    for scale in [0.25, 0.5, 0.75, 1.25, 1.5, 2.0, 3.0]
        t0 = time()
        g_val = 75.0 * scale

        # Voltage-driven instability scenario
        m_v  = SatoBersModel{Float64}(tauf=TAUF_V,  av=U_V,  g_rel=g_val, N_CaL=100000)
        mu_v, sig_v = run_condition(m_v, prepace=400)

        # Ca²⁺-driven instability scenario
        m_ca = SatoBersModel{Float64}(tauf=TAUF_CA, av=U_CA, g_rel=g_val, N_CaL=100000)
        mu_ca, sig_ca = run_condition(m_ca, prepace=400)

        println(io, "$scale,$mu_v,$sig_v,$mu_ca,$sig_ca")
        elapsed = round(time() - t0, digits=1)
        println("  GRyR×$(scale): V-driven APD_std=$(round(sig_v,digits=3)), Ca-driven APD_std=$(round(sig_ca,digits=3)) ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/pharm_grel_sweep.csv")

# ============================================================================
# 2. GKr sweep — Fig A3B / Fig 6
# ============================================================================
println("\n=== GKr sweep (Fig A3B / 6) ===")
open(joinpath(@__DIR__, "data", "pharm_gkr_sweep.csv"), "w") do io
    println(io, "gkr_scale,apd_mean_v,apd_std_v,apd_mean_ca,apd_std_ca")
    println(io, "1.0,$(mu_v_ctrl),$(sig_v_ctrl),$(mu_ca_ctrl),$(sig_ca_ctrl)")

    for scale in [0.25, 0.5, 0.75, 1.25, 1.5, 2.0]
        t0 = time()
        gkr_val = 0.0136 * scale

        m_v  = SatoBersModel{Float64}(tauf=TAUF_V,  av=U_V,  gkr=gkr_val, N_CaL=100000)
        mu_v, sig_v = run_condition(m_v, prepace=400)

        m_ca = SatoBersModel{Float64}(tauf=TAUF_CA, av=U_CA, gkr=gkr_val, N_CaL=100000)
        mu_ca, sig_ca = run_condition(m_ca, prepace=400)

        println(io, "$scale,$mu_v,$sig_v,$mu_ca,$sig_ca")
        elapsed = round(time() - t0, digits=1)
        println("  GKr×$(scale): V-driven APD_std=$(round(sig_v,digits=3)), Ca-driven APD_std=$(round(sig_ca,digits=3)) ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/pharm_gkr_sweep.csv")

# ============================================================================
# 3. GKs sweep — Fig A3C / Fig 7
# ============================================================================
println("\n=== GKs sweep (Fig A3C / 7) ===")
open(joinpath(@__DIR__, "data", "pharm_gks_sweep.csv"), "w") do io
    println(io, "gks_scale,apd_mean_v,apd_std_v,apd_mean_ca,apd_std_ca")
    println(io, "1.0,$(mu_v_ctrl),$(sig_v_ctrl),$(mu_ca_ctrl),$(sig_ca_ctrl)")

    for scale in [0.25, 0.5, 0.75, 1.25, 1.5, 2.0]
        t0 = time()
        gks_val = 0.0245 * scale

        m_v  = SatoBersModel{Float64}(tauf=TAUF_V,  av=U_V,  gks=gks_val, N_CaL=100000)
        mu_v, sig_v = run_condition(m_v, prepace=400)

        m_ca = SatoBersModel{Float64}(tauf=TAUF_CA, av=U_CA, gks=gks_val, N_CaL=100000)
        mu_ca, sig_ca = run_condition(m_ca, prepace=400)

        println(io, "$scale,$mu_v,$sig_v,$mu_ca,$sig_ca")
        elapsed = round(time() - t0, digits=1)
        println("  GKs×$(scale): V-driven APD_std=$(round(sig_v,digits=3)), Ca-driven APD_std=$(round(sig_ca,digits=3)) ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/pharm_gks_sweep.csv")

# ============================================================================
# 4. GCaL sweep (icabar scaling) — Fig 8
# ============================================================================
println("\n=== GCaL sweep (Fig 8) ===")
open(joinpath(@__DIR__, "data", "pharm_gcal_sweep.csv"), "w") do io
    println(io, "gcal_scale,apd_mean_v,apd_std_v,apd_mean_ca,apd_std_ca")
    println(io, "1.0,$(mu_v_ctrl),$(sig_v_ctrl),$(mu_ca_ctrl),$(sig_ca_ctrl)")

    for scale in [0.5, 0.6, 0.7, 0.8, 0.9, 1.1, 1.2, 1.3, 1.5]
        t0 = time()
        icabar_val = 3.5 * scale

        m_v  = SatoBersModel{Float64}(tauf=TAUF_V,  av=U_V,  icabar=icabar_val, N_CaL=100000)
        mu_v, sig_v = run_condition(m_v, prepace=400)

        m_ca = SatoBersModel{Float64}(tauf=TAUF_CA, av=U_CA, icabar=icabar_val, N_CaL=100000)
        mu_ca, sig_ca = run_condition(m_ca, prepace=400)

        println(io, "$scale,$mu_v,$sig_v,$mu_ca,$sig_ca")
        elapsed = round(time() - t0, digits=1)
        println("  GCaL×$(scale): V-driven APD_std=$(round(sig_v,digits=3)), Ca-driven APD_std=$(round(sig_ca,digits=3)) ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/pharm_gcal_sweep.csv")

# ============================================================================
# 5. GNa sweep — Fig 9/10
# ============================================================================
println("\n=== GNa sweep (Fig 9/10) ===")
open(joinpath(@__DIR__, "data", "pharm_gna_sweep.csv"), "w") do io
    println(io, "gna_scale,apd_mean_v,apd_std_v,apd_mean_ca,apd_std_ca")
    println(io, "1.0,$(mu_v_ctrl),$(sig_v_ctrl),$(mu_ca_ctrl),$(sig_ca_ctrl)")

    for scale in [0.5, 0.75, 1.25, 1.5, 2.0, 3.0]
        t0 = time()
        gna_val = 12.0 * scale

        m_v  = SatoBersModel{Float64}(tauf=TAUF_V,  av=U_V,  gna=gna_val, N_CaL=100000)
        mu_v, sig_v = run_condition(m_v, prepace=400)

        m_ca = SatoBersModel{Float64}(tauf=TAUF_CA, av=U_CA, gna=gna_val, N_CaL=100000)
        mu_ca, sig_ca = run_condition(m_ca, prepace=400)

        println(io, "$scale,$mu_v,$sig_v,$mu_ca,$sig_ca")
        elapsed = round(time() - t0, digits=1)
        println("  GNa×$(scale): V-driven APD_std=$(round(sig_v,digits=3)), Ca-driven APD_std=$(round(sig_ca,digits=3)) ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/pharm_gna_sweep.csv")

println("\n=== All pharmacological sweep data generated ===")
println("Run figures/plot_pharm_figures.py to generate plots.")
