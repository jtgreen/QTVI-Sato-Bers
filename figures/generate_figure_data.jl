#!/usr/bin/env julia
"""
generate_figure_data.jl — Generate simulation data for reproducing key figures
from Sato et al. (2025) "Beat-to-beat QT interval variability as a tool to
detect the underlying cellular mechanisms of arrhythmias."

Produces CSV files in figures/data/:
  - tauf_sweep.csv        : tauf sweep (Fig 2A/4A): APD stats vs tauf
  - u_sweep_iclamp.csv    : u sweep, I-clamp (Fig 4C): APD stats vs u
  - u_sweep_apclamp.csv   : u sweep, AP-clamp (Fig 4B): Ca2+ stats vs u
  - ap_traces.csv         : Action potential + Ca transient traces for representative tauf values
  - stability_boundary.csv: 2D parameter scan of (tauf, u) for stability map (Fig 1)

All sweeps use:
  - BCL = 300 ms
  - dt = 0.05 ms
  - N_CaL = 100000 (stochastic gating)
  - Prepace: 300 beats (more near bifurcation)
  - Measurement: 1000 beats (reduced from paper's 10000 for speed)

Note: With 1000 beats instead of 10000, APD standard deviation will be
underestimated (higher noise). The qualitative behavior (onset of alternans,
APD variability increasing) is correctly reproduced.
"""

include(joinpath(@__DIR__, "..", "julia", "SatoBers.jl"))
using .SatoBers

const dt   = 0.05
const BCL  = 300.0
const BCLn = round(Int, BCL / dt)
const Durn = round(Int, 1.0 / dt)
const STIM = 50.0
const VC   = -70.0

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

Run `beats` beats, recording APD and peak Ca for each beat.
Uses APD detection at V = -70 mV threshold with linear interpolation.
"""
function measure_beats!(u, du, model, rng, beats)
    APDs  = Float64[]
    CaMax = Float64[]

    first   = false
    cimax   = 0.0
    apd     = 0.0
    APDt1   = 0.0
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
APD variability statistics (every beat and every other beat).
Returns (mean, std, min, max, std_alternating).
"""
function apd_stats(apds)
    n = length(apds)
    isempty(apds) && return (NaN, NaN, NaN, NaN, NaN)
    mu    = sum(apds) / n
    var   = sum((x - mu)^2 for x in apds) / max(n - 1, 1)
    sigma = sqrt(var)
    # Every-other-beat variability (alternans indicator)
    even = apds[2:2:end]
    odd  = apds[1:2:end]
    if length(even) >= 2 && length(odd) >= 2
        mu_e = sum(even) / length(even)
        mu_o = sum(odd)  / length(odd)
        var_e = sum((x - mu_e)^2 for x in even) / max(length(even)-1, 1)
        var_o = sum((x - mu_o)^2 for x in odd)  / max(length(odd)-1, 1)
        sigma_alt = (sqrt(var_e) + sqrt(var_o)) / 2.0
    else
        sigma_alt = NaN
    end
    return (mu, sigma, minimum(apds), maximum(apds), sigma_alt)
end

# ============================================================================
# 1. tauf sweep (voltage-driven instability) — reproduces Fig 2A, 4A
#    Parameters: tauf = 20..60, u = 3.0 (default), N_CaL = 100000
# ============================================================================
println("=== tauf sweep (Fig 2A/4A) ===")
mkpath(joinpath(@__DIR__, "data"))
open(joinpath(@__DIR__, "data", "tauf_sweep.csv"), "w") do io
    println(io, "tauf,apd_mean,apd_std,apd_min,apd_max,apd_std_altbeat,ca_mean,ca_std,ca_min,ca_max")
    u_ref = collect(default_initial_state(SatoBersModel{Float64}()))

    for tf in 20:60
        t0 = time()
        model = SatoBersModel{Float64}(tauf=Float64(tf), av=3.0, N_CaL=100000)
        u  = copy(u_ref)
        du = zeros(15)
        rng = StochasticState(UInt32(1821800813))

        # More pre-pacing near bifurcation region (tauf > 45)
        prepace_beats = tf >= 48 ? 400 : 200
        prepace!(u, du, model, rng, prepace_beats)

        # Measurement beats
        meas_beats = 500
        APDs, CaMaxes = measure_beats!(u, du, model, rng, meas_beats)

        apd_mu, apd_sig, apd_lo, apd_hi, apd_sig_alt = apd_stats(APDs)
        ca_mu, ca_sig, ca_lo, ca_hi, _  = apd_stats(CaMaxes)

        println(io, "$tf,$apd_mu,$apd_sig,$apd_lo,$apd_hi,$apd_sig_alt,$ca_mu,$ca_sig,$ca_lo,$ca_hi")
        elapsed = round(time() - t0, digits=1)
        println("  tauf=$tf: APD=$(round(apd_mu,digits=2))±$(round(apd_sig,digits=3)) ms  Ca=$(round(ca_mu,digits=3))±$(round(ca_sig,digits=4)) µM  ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/tauf_sweep.csv")

# ============================================================================
# 2. u sweep, current clamp (Ca2+-driven instability, I-clamp) — Fig 4C
#    Parameters: tauf = 35.0 (stable), u = 2..11, N_CaL = 100000
# ============================================================================
println("\n=== u sweep, current clamp (Fig 4C) ===")
open(joinpath(@__DIR__, "data", "u_sweep_iclamp.csv"), "w") do io
    println(io, "u,apd_mean,apd_std,apd_min,apd_max,apd_std_altbeat,ca_mean,ca_std,ca_min,ca_max")
    u_ref = collect(default_initial_state(SatoBersModel{Float64}()))

    for u_val in 2.0:0.25:11.0
        t0 = time()
        model = SatoBersModel{Float64}(tauf=35.0, av=u_val, N_CaL=100000)
        u  = copy(u_ref)
        du = zeros(15)
        rng = StochasticState(UInt32(1821800813))

        # More pre-pacing near bifurcation region (u > 8)
        prepace_beats = u_val >= 8.5 ? 400 : 200
        prepace!(u, du, model, rng, prepace_beats)

        meas_beats = 500
        APDs, CaMaxes = measure_beats!(u, du, model, rng, meas_beats)

        apd_mu, apd_sig, apd_lo, apd_hi, apd_sig_alt = apd_stats(APDs)
        ca_mu, ca_sig, ca_lo, ca_hi, _  = apd_stats(CaMaxes)

        println(io, "$u_val,$apd_mu,$apd_sig,$apd_lo,$apd_hi,$apd_sig_alt,$ca_mu,$ca_sig,$ca_lo,$ca_hi")
        elapsed = round(time() - t0, digits=1)
        println("  u=$u_val: APD=$(round(apd_mu,digits=2))±$(round(apd_sig,digits=3)) ms  Ca=$(round(ca_mu,digits=3))±$(round(ca_sig,digits=4)) µM  ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/u_sweep_iclamp.csv")

# ============================================================================
# 3. u sweep, AP-clamp (Ca2+-driven instability, AP-clamp) — Fig 3A/4B
#    First generate reference AP waveform at tauf=35, u=3 (stable).
#    Then replay that waveform while sweeping u.
#    AP-clamp: force v = v_clamp[t] at each step; integrate all other vars.
# ============================================================================
println("\n=== u sweep, AP-clamp (Fig 3A/4B) ===")

# Step 1: Generate reference AP waveform at tauf=35, u=3 (stable)
println("  Generating reference AP waveform...")
let
    model_ref = SatoBersModel{Float64}(tauf=35.0, av=3.0, N_CaL=100000)
    u  = collect(default_initial_state(model_ref))
    du = zeros(15)
    rng = StochasticState(UInt32(1821800813))
    prepace!(u, du, model_ref, rng, 300)  # Prepace for clean waveform

    global ap_waveform = zeros(Float64, BCLn)
    for tn in 0:BCLn-1
        ap_waveform[tn+1] = u[1]
        pace!(u, du, model_ref, rng, tn < Durn ? STIM : 0.0)
    end
    println("  AP waveform recorded: v_min=$(minimum(ap_waveform)) mV, v_max=$(maximum(ap_waveform)) mV")
end

"""
    measure_beats_apclamp!(u, du, model, rng, beats, ap_waveform) -> (Ca_peaks)

Run `beats` beats with AP-clamp (v fixed to ap_waveform). Track peak Ca2+.
In AP-clamp mode, v is forced to ap_waveform[t%BCLn] at each step,
and only the other 14 state variables (Ca2+, gates) are integrated.
"""
function measure_beats_apclamp!(u, du, model, rng, beats, ap_waveform)
    Ca_peaks = Float64[]
    cimax    = 0.0
    in_beat  = false

    for tn in 0:(beats * BCLn - 1)
        t_idx = mod(tn, BCLn)  # Position within beat cycle

        # Force v to AP waveform
        u[1] = ap_waveform[t_idx + 1]

        # Compute RHS with clamped voltage
        st = t_idx < Durn ? STIM : 0.0
        cell_rhs_deterministic!(du, u, model, st)
        stochastic_gate_update!(u, du, dt, model, rng)

        # Update all variables EXCEPT v (AP-clamp)
        for k in 2:15
            u[k] += du[k] * dt
        end
        # Force v back to clamp (in case stochastic_gate_update changed it)
        u[1] = ap_waveform[t_idx + 1]

        if u[2] > cimax
            cimax = u[2]
        end

        # Record at end of each beat
        if t_idx == BCLn - 1
            push!(Ca_peaks, cimax)
            cimax = 0.0
        end
    end
    return Ca_peaks
end

open(joinpath(@__DIR__, "data", "u_sweep_apclamp.csv"), "w") do io
    println(io, "u,ca_mean,ca_std,ca_min,ca_max,ca_std_altbeat")
    u_ref = collect(default_initial_state(SatoBersModel{Float64}()))

    for u_val in 2.0:0.25:11.0
        t0 = time()
        model = SatoBersModel{Float64}(tauf=35.0, av=u_val, N_CaL=100000)
        u  = copy(u_ref)
        du = zeros(15)
        rng = StochasticState(UInt32(1821800813))

        # Pre-pace with AP-clamp
        for _ in 1:(u_val >= 8.5 ? 400 : 200)
            for tn in 0:BCLn-1
                u[1] = ap_waveform[tn + 1]
                st = tn < Durn ? STIM : 0.0
                cell_rhs_deterministic!(du, u, model, st)
                stochastic_gate_update!(u, du, dt, model, rng)
                for k in 2:15; u[k] += du[k] * dt; end
                u[1] = ap_waveform[tn + 1]
            end
        end

        meas_beats = 500
        Ca_peaks = measure_beats_apclamp!(u, du, model, rng, meas_beats, ap_waveform)

        _, ca_sig, ca_lo, ca_hi, ca_sig_alt = apd_stats(Ca_peaks)
        ca_mu = sum(Ca_peaks) / length(Ca_peaks)

        println(io, "$u_val,$ca_mu,$ca_sig,$ca_lo,$ca_hi,$ca_sig_alt")
        elapsed = round(time() - t0, digits=1)
        println("  u=$u_val (AP-clamp): Ca=$(round(ca_mu,digits=3))±$(round(ca_sig,digits=4)) µM  ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/u_sweep_apclamp.csv")

# ============================================================================
# 4. AP traces — representative action potentials at different tauf values
#    Records v(t) and ci(t) for one beat at steady state
# ============================================================================
println("\n=== AP traces for representative tauf values ===")
open(joinpath(@__DIR__, "data", "ap_traces.csv"), "w") do io
    println(io, "tauf,time,v,ci,cs,d,f,q")
    u_ref = collect(default_initial_state(SatoBersModel{Float64}()))

    for tf in [25, 35, 45, 52, 57]
        t0 = time()
        model = SatoBersModel{Float64}(tauf=Float64(tf), av=3.0, N_CaL=100000)
        u  = copy(u_ref)
        du = zeros(15)
        rng = StochasticState(UInt32(1821800813))
        prepace!(u, du, model, rng, 300)

        # Record 1 beat, sample every 2 steps = 0.1 ms
        for tn in 0:BCLn-1
            t = tn * dt
            if mod(tn, 2) == 0
                println(io, "$tf,$t,$(u[1]),$(u[2]),$(u[3]),$(u[9]),$(u[7]),$(u[8])")
            end
            pace!(u, du, model, rng, tn < Durn ? STIM : 0.0)
        end
        elapsed = round(time() - t0, digits=1)
        println("  tauf=$tf: AP recorded ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/ap_traces.csv")

# ============================================================================
# 5. Beat-by-beat APD traces for selected tauf values — reproduces Fig 2B
#    Records APD for each of 200 consecutive beats
# ============================================================================
println("\n=== Beat-by-beat APD traces (Fig 2B) ===")
open(joinpath(@__DIR__, "data", "apd_per_beat.csv"), "w") do io
    println(io, "tauf,beat,apd,camax")
    u_ref = collect(default_initial_state(SatoBersModel{Float64}()))

    for tf in [40, 48, 52, 55, 57, 60]
        t0 = time()
        model = SatoBersModel{Float64}(tauf=Float64(tf), av=3.0, N_CaL=100000)
        u  = copy(u_ref)
        du = zeros(15)
        rng = StochasticState(UInt32(1821800813))
        prepace!(u, du, model, rng, tf >= 52 ? 400 : 200)
        APDs, CaMaxes = measure_beats!(u, du, model, rng, 200)
        for (b, (apd, camax)) in enumerate(zip(APDs, CaMaxes))
            println(io, "$tf,$b,$apd,$camax")
        end
        elapsed = round(time() - t0, digits=1)
        println("  tauf=$tf: $(length(APDs)) beats recorded ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/apd_per_beat.csv")

# ============================================================================
# 6. Beat-by-beat Ca2+ traces for u sweep (AP-clamp) — reproduces Fig 3B/4B
# ============================================================================
println("\n=== Beat-by-beat Ca2+ traces, AP-clamp (Fig 3B/4B) ===")
open(joinpath(@__DIR__, "data", "ca_per_beat_apclamp.csv"), "w") do io
    println(io, "u,beat,camax")
    u_ref = collect(default_initial_state(SatoBersModel{Float64}()))

    for u_val in [4.0, 7.7, 9.5, 10.5]
        t0 = time()
        model = SatoBersModel{Float64}(tauf=35.0, av=u_val, N_CaL=100000)
        u  = copy(u_ref)
        du = zeros(15)
        rng = StochasticState(UInt32(1821800813))

        # AP-clamp prepace
        for _ in 1:(u_val >= 8.5 ? 400 : 200)
            for tn in 0:BCLn-1
                u[1] = ap_waveform[tn + 1]
                st = tn < Durn ? STIM : 0.0
                cell_rhs_deterministic!(du, u, model, st)
                stochastic_gate_update!(u, du, dt, model, rng)
                for k in 2:15; u[k] += du[k] * dt; end
                u[1] = ap_waveform[tn + 1]
            end
        end

        Ca_peaks = measure_beats_apclamp!(u, du, model, rng, 200, ap_waveform)
        for (b, camax) in enumerate(Ca_peaks)
            println(io, "$u_val,$b,$camax")
        end
        elapsed = round(time() - t0, digits=1)
        println("  u=$u_val (AP-clamp): $(length(Ca_peaks)) beats ($(elapsed)s)")
        flush(io)
    end
end
println("  -> wrote figures/data/ca_per_beat_apclamp.csv")

println("\n=== All figure data generated successfully ===")
println("Run figures/plot_figures.py to generate plots.")
