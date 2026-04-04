#!/usr/bin/env julia
"""
tissue_armyheart_batch.jl — Multi-GPU parameter sweep using ArmyHeart's batch.jl

Distributes the τ_f and u sweeps (Figs A2C, A2D) across multiple A30 GPUs.
Each GPU runs one (τ_f or u) parameter point independently.

Usage:
    julia --project=/path/to/ArmyHeart tissue_armyheart_batch.jl

Requires:
  - CUDA.jl (for GPU access)
  - ArmyHeart (for batch.jl)
  - 2+ CUDA-capable GPUs

Environment variables:
  ARMYHEART_PATH  — path to ArmyHeart checkout (default: /tmp/ArmyHeart)
  SATO_N_GPU      — number of GPUs to use (default: all available)
"""

using Distributed

const ARMYHEART_PATH = get(ENV, "ARMYHEART_PATH", "/tmp/ArmyHeart")
const SCRIPT_DIR     = @__DIR__

# Add one worker per GPU
using CUDA
n_gpus     = parse(Int, get(ENV, "SATO_N_GPU", string(length(CUDA.devices()))))
n_gpus_use = min(n_gpus, 9)   # cap at 9 A30s
addprocs(n_gpus_use)

@everywhere begin
    using Pkg
    const _ARMYHEART = $(ARMYHEART_PATH)
    const _SCRIPT    = $(SCRIPT_DIR)
    if isdir(_ARMYHEART)
        Pkg.activate(_ARMYHEART; io=devnull)
        Pkg.instantiate(; io=devnull)
    end

    using CUDA
    using Statistics
    using Printf

    include(joinpath(_SCRIPT, "SatoArmyHeart.jl"))

    # Inline the simulation function (avoids file re-include on workers)
    include(joinpath(_SCRIPT, "tissue_armyheart.jl"))   # defines run_tissue_sim etc.
end

# ─────────────────────────────────────────────────────────────────────────────
# Parameter sweep definitions
# ─────────────────────────────────────────────────────────────────────────────

tauf_sweep_params = [
    (tauf=tf, av=3.0, label="tauf_$(round(Int,tf))")
    for tf in [20.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 52.0]
]

u_sweep_params = [
    (tauf=35.0, av=uv, label="u_$(uv)")
    for uv in [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5]
]

all_params = vcat(tauf_sweep_params, u_sweep_params)

# ─────────────────────────────────────────────────────────────────────────────
# Batch runner (mirrors ArmyHeart's batch.jl pattern)
# ─────────────────────────────────────────────────────────────────────────────

"""
Run one tissue simulation on the given GPU device.
Returns (label, qt_mean, qt_std, qtvi, n_beats).
"""
function simulate_one(params; device_id::Int=0)
    CUDA.device!(device_id)
    ENV["SATO_USE_GPU"] = "true"

    @printf("  [GPU %d] running %s...\n", device_id, params.label)
    qts, ecg = run_tissue_sim(;
        tauf          = params.tauf,
        av            = params.av,
        N_CaL         = 100000,
        prepace_beats = 20,
        meas_beats    = 60,
        verbose       = false,
    )
    if length(qts) < 5
        return (params.label, NaN, NaN, NaN, length(qts))
    end
    μ = mean(qts); σ = std(qts)
    qtvi = compute_qtvi(qts)
    @printf("  [GPU %d] %s done: QT=%.2f±%.3f QTVi=%.4f (n=%d)\n",
            device_id, params.label, μ, σ, qtvi, length(qts))
    return (params.label, μ, σ, qtvi, length(qts))
end

# ─────────────────────────────────────────────────────────────────────────────
# Distribute across GPUs using ArmyHeart's batch pattern
# ─────────────────────────────────────────────────────────────────────────────
function run_batch(params_list, n_gpus)
    device_ids = 0:n_gpus-1
    tasks_with_device = [(i % n_gpus, p) for (i, p) in enumerate(params_list)]

    results = asyncmap(zip(workers(), Iterators.cycle(device_ids))) do (wid, dev)
        my_tasks = filter(t -> t[1] == dev, tasks_with_device)
        map(my_tasks) do (_, p)
            remotecall_fetch(wid) do
                simulate_one(p; device_id=dev)
            end
        end
    end
    return vcat(results...)
end

println("="^70)
println("Tissue Batch Sweep — ArmyHeart multi-GPU")
println("GPUs: $n_gpus_use  |  Parameter points: $(length(all_params))")
println("="^70)

results = run_batch(all_params, n_gpus_use)

# ─────────────────────────────────────────────────────────────────────────────
# Write results
# ─────────────────────────────────────────────────────────────────────────────
out = joinpath(@__DIR__, "..", "figures", "data")
mkpath(out)

# Split results back into tauf and u sweeps
tauf_results = filter(r -> startswith(r[1], "tauf"), results)
u_results    = filter(r -> startswith(r[1], "u_"),   results)

open(joinpath(out, "tissue2d_tauf_qtv.csv"), "w") do io
    println(io, "tauf,qt_mean,qt_std,qtvi,n_beats")
    for r in sort(tauf_results, by=x->x[1])
        tf = parse(Float64, split(r[1], "_")[end])
        println(io, "$tf,$(r[2]),$(r[3]),$(r[4]),$(r[5])")
    end
end

open(joinpath(out, "tissue2d_u_qtv.csv"), "w") do io
    println(io, "u,qt_mean,qt_std,qtvi,n_beats")
    for r in sort(u_results, by=x->x[1])
        uv = parse(Float64, split(r[1], "_")[end])
        println(io, "$uv,$(r[2]),$(r[3]),$(r[4]),$(r[5])")
    end
end

println("\n✓ Batch sweep complete.")
println("  → tissue2d_tauf_qtv.csv")
println("  → tissue2d_u_qtv.csv")
