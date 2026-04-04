#!/usr/bin/env julia
# Quick validation of Julia SatoBers port against C++ reference traces.
# Runs scenarios 1, 2, 4a-d, 5, 6 (skips slow scenario 3).

include(joinpath(@__DIR__, "..", "julia", "SatoBers.jl"))
using .SatoBers

function load_csv(fn)
    lines = readlines(fn)
    n = length(lines) - 1
    ncol = length(split(lines[1], ","))
    data = zeros(Float64, n, ncol)
    for i in 1:n
        vals = split(lines[i+1], ",")
        for j in 1:ncol
            data[i, j] = parse(Float64, vals[j])
        end
    end
    return data
end

function compare_traces(ref_data, jl_data; atol=1e-12)
    n = min(size(ref_data, 1), size(jl_data, 1))
    max_abs = 0.0
    max_rel = 0.0
    for i in 1:n
        for j in 2:min(size(ref_data, 2), size(jl_data, 2))
            rv = ref_data[i, j]
            jv = jl_data[i, j]
            ae = abs(rv - jv)
            denom = max(abs(rv), abs(jv), 1e-300)
            max_abs = max(max_abs, ae)
            max_rel = max(max_rel, ae / denom)
        end
    end
    passed = (max_abs < atol) || (max_rel < atol)
    return passed, max_abs, max_rel
end

function prepace!(u, du, model, beats; rng=nothing)
    dt = 0.05
    BCLn = round(Int, 300.0 / dt)
    Durn = round(Int, 1.0 / dt)
    for _ in 1:beats
        for tn in 0:(BCLn - 1)
            st = tn < Durn ? 50.0 : 0.0
            cell_rhs_deterministic!(du, u, model, st)
            if rng !== nothing
                stochastic_gate_update!(u, du, dt, model, rng)
            end
            @. u += du * dt
        end
    end
end

function record_beat!(u, du, model, si; rng=nothing)
    dt = 0.05
    BCLn = round(Int, 300.0 / dt)
    Durn = round(Int, 1.0 / dt)
    nsamples = div(BCLn, si)
    trace = zeros(Float64, nsamples, 16)
    idx = 0
    for tn in 0:(BCLn - 1)
        t = tn * dt
        if mod(tn, si) == 0
            idx += 1
            if idx <= nsamples
                trace[idx, 1] = t
                trace[idx, 2:16] .= u[1:15]
            end
        end
        st = tn < Durn ? 50.0 : 0.0
        cell_rhs_deterministic!(du, u, model, st)
        if rng !== nothing
            stochastic_gate_update!(u, du, dt, model, rng)
        end
        @. u += du * dt
    end
    return trace[1:idx, :]
end

function scenario1()
    println("Scenario 1: Single beat deterministic")
    t0 = time()
    m = SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0)
    u = collect(default_initial_state(m))
    du = zeros(15)
    prepace!(u, du, m, 100)
    trace = record_beat!(u, du, m, 10)
    ref = load_csv(joinpath(@__DIR__, "reference", "scenario1_single_beat.csv"))
    ok, a, r = compare_traces(ref, trace; atol=1e-12)
    println("  $(ok ? "PASS" : "FAIL"): max_abs=$(round(a, sigdigits=3))  max_rel=$(round(r, sigdigits=3))  ($(round(time()-t0, digits=1))s)")
    return ok
end

function scenario2()
    println("Scenario 2: Multi-beat convergence")
    t0 = time()
    m = SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0)
    u = collect(default_initial_state(m))
    du = zeros(15)
    prepace!(u, du, m, 100)
    dt = 0.05
    BCLn = round(Int, 300.0 / dt)
    Durn = round(Int, 1.0 / dt)
    total = 10 * BCLn
    nsamples = div(total, 20)
    trace = zeros(Float64, nsamples, 16)
    idx = 0
    for tn in 0:(total - 1)
        t = tn * dt
        if mod(tn, 20) == 0
            idx += 1
            if idx <= nsamples
                trace[idx, 1] = t
                trace[idx, 2:16] .= u[1:15]
            end
        end
        st = mod(tn, BCLn) < Durn ? 50.0 : 0.0
        cell_rhs_deterministic!(du, u, m, st)
        @. u += du * dt
    end
    trace = trace[1:idx, :]
    ref = load_csv(joinpath(@__DIR__, "reference", "scenario2_multi_beat.csv"))
    ok, a, r = compare_traces(ref, trace; atol=1e-12)
    println("  $(ok ? "PASS" : "FAIL"): max_abs=$(round(a, sigdigits=3))  max_rel=$(round(r, sigdigits=3))  ($(round(time()-t0, digits=1))s)")
    return ok
end

function scenario4()
    println("Scenario 4: Perturbation tests")
    all_ok = true
    for (label, fname, mod_) in [
        ("4a: 2x gk1",   "scenario4a_elevated_gk1.csv",  SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0, gk1=5.6)),
        ("4b: 50% ICaL", "scenario4b_ical_block.csv",     SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0, icabar=1.75)),
        ("4c: 2x SERCA", "scenario4c_enhanced_serca.csv", SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0, vup=0.500)),
    ]
        u_ = collect(default_initial_state(mod_))
        du_ = zeros(15)
        prepace!(u_, du_, mod_, 100)
        trace_ = record_beat!(u_, du_, mod_, 10)
        ref_ = load_csv(joinpath(@__DIR__, "reference", fname))
        ok_, a_, r_ = compare_traces(ref_, trace_; atol=1e-11)
        println("  $label: $(ok_ ? "PASS" : "FAIL") max_abs=$(round(a_, sigdigits=3))  max_rel=$(round(r_, sigdigits=3))")
        all_ok = all_ok && ok_
    end
    # 4d: depolarized
    m4d = SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0)
    u = collect(default_initial_state(m4d))
    u[1] = -60.0
    du = zeros(15)
    prepace!(u, du, m4d, 100)
    trace = record_beat!(u, du, m4d, 10)
    ref = load_csv(joinpath(@__DIR__, "reference", "scenario4d_depolarized.csv"))
    ok, a, r = compare_traces(ref, trace; atol=1e-11)
    println("  4d: v0=-60mV: $(ok ? "PASS" : "FAIL") max_abs=$(round(a, sigdigits=3))  max_rel=$(round(r, sigdigits=3))")
    all_ok = all_ok && ok
    return all_ok
end

function scenario5()
    println("Scenario 5: Stochastic gating (fixed seed)")
    t0 = time()
    m = SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=100000)
    u = collect(default_initial_state(m))
    du = zeros(15)
    rng = StochasticState(UInt32(1821800813))
    prepace!(u, du, m, 50; rng=rng)
    trace = record_beat!(u, du, m, 10; rng=rng)
    ref = load_csv(joinpath(@__DIR__, "reference", "scenario5_stochastic.csv"))
    ok, a, r = compare_traces(ref, trace; atol=1e-10)
    println("  $(ok ? "PASS" : "FAIL"): max_abs=$(round(a, sigdigits=3))  max_rel=$(round(r, sigdigits=3))  ($(round(time()-t0, digits=1))s)")
    return ok
end

function scenario6()
    println("Scenario 6: Steady-state fingerprint (500 beats)")
    t0 = time()
    m = SatoBersModel{Float64}(tauf=30.0, av=3.0, N_CaL=0)
    u = collect(default_initial_state(m))
    du = zeros(15)
    prepace!(u, du, m, 500)
    jl_row = zeros(Float64, 1, 16)
    jl_row[1, 2:16] .= u[1:15]
    ref = load_csv(joinpath(@__DIR__, "reference", "scenario6_steady_state.csv"))
    ok, a, r = compare_traces(ref, jl_row; atol=1e-12)
    println("  $(ok ? "PASS" : "FAIL"): max_abs=$(round(a, sigdigits=3))  max_rel=$(round(r, sigdigits=3))  ($(round(time()-t0, digits=1))s)")
    return ok
end

function main()
    println("=" ^ 60)
    println("Sato-Bers Model: Julia Quick Validation Suite")
    println("=" ^ 60)
    println()

    results = Dict{String, Bool}(
        "scenario1" => scenario1(),
        "scenario2" => scenario2(),
        "scenario4" => scenario4(),
        "scenario5" => scenario5(),
        "scenario6" => scenario6(),
    )

    println()
    println("=" ^ 60)
    println("SUMMARY")
    println("=" ^ 60)
    all_passed = true
    for k in sort(collect(keys(results)))
        status = results[k] ? "PASS" : "FAIL"
        println("  $k: $status")
        if !results[k]
            all_passed = false
        end
    end
    println()
    if all_passed
        println("ALL SCENARIOS PASSED")
    else
        println("SOME SCENARIOS FAILED")
    end
    return all_passed ? 0 : 1
end

exit(main())
