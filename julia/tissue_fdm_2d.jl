#!/usr/bin/env julia
"""
tissue_fdm_2d.jl — 2D monodomain tissue simulation using ArmyHeart FDM kernel

Reproduces Sato et al. 2025 Appendix Fig A2:
  - 3 cm × 3 cm, 61×61 = 3721 nodes (60×60 elements, dx = 0.5 mm)
  - Corner pacing (bottom-left region)
  - Pseudo-ECG at opposite corner (Plonsey 1964)
  - QTVi vs τ_f (Fig A2C) and QTVi vs u (Fig A2D)
  - Stochastic ICaL gating (N_CaL = 100,000)

Architecture:
  - ArmyHeart FDM: LazyGrid + ∇²_2d! CUDA kernel (orthogonal geometry speedup)
  - SatoArmyHeart.jl: SatoBersArmyModel (our ionic model)
  - Custom KernelAbstractions cell kernel for GPU parallelism
  - Lie-Trotter operator splitting: cell step → diffusion step
  - Float64 on A30 GPU

Run modes:
  julia tissue_fdm_2d.jl                    # CPU Float64
  SATO_USE_GPU=true julia tissue_fdm_2d.jl  # A30 GPU Float64

Usage:
  # Single run (test)
  julia tissue_fdm_2d.jl

  # Full parameter sweep (tauf + u)
  SATO_FULL_SWEEP=true julia tissue_fdm_2d.jl
"""

# ============================================================
# 0. GPU / CPU setup
# ============================================================
const USE_GPU = parse(Bool, get(ENV, "SATO_USE_GPU", "false"))
const FULL_SWEEP = parse(Bool, get(ENV, "SATO_FULL_SWEEP", "false"))
const ARMYHEART_FDM = get(ENV, "ARMYHEART_FDM_SRC",
    joinpath(@__DIR__, "armyheart_fdm"))

if USE_GPU
    using CUDA
    @info "GPU mode: $(CUDA.name(CUDA.device()))"
end

import KernelAbstractions as KA
using StaticArrays
using LinearAlgebra
using Statistics
using Printf
using Thunderbolt   # for Vec type + cell_rhs! interface

# Include ArmyHeart FDM components
include(joinpath(ARMYHEART_FDM, "grid.jl"))     # LazyGrid

# Include our SatoBers ArmyHeart cell model
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================
# 1. ArmyHeart FDM Kernel (5-point stencil, GPU-compatible)
#    Copied from ArmyHeart finite-diff-method branch
#    src/electrophysiology/fdm/kernels.jl
# ============================================================
@inline idx2flat(i, j, Nx) = (j - 1) * Nx + i

KA.@kernel function ∇²_2d_kernel!(d2u, u, t, κ, stim, dx, dy, Nx, Ny)
    i, j = KA.@index(Global, NTuple)
    idx = (j - 1) * Nx + i

    pos = Thunderbolt.Vec(((i - 1) * dx, (j - 1) * dy))
    I_stim = stim(pos, t)

    if 2 <= i <= Nx - 1 && 2 <= j <= Ny - 1
        d2x = (u[idx2flat(i+1,j,Nx)] - 2u[idx] + u[idx2flat(i-1,j,Nx)]) / (dx^2)
        d2y = (u[idx2flat(i,j+1,Nx)] - 2u[idx] + u[idx2flat(i,j-1,Nx)]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == 1 && 2 <= j <= Ny - 1
        d2x = 2(u[idx2flat(i+1,j,Nx)] - u[idx]) / (dx^2)
        d2y = (u[idx2flat(i,j+1,Nx)] - 2u[idx] + u[idx2flat(i,j-1,Nx)]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == Nx && 2 <= j <= Ny - 1
        d2x = 2(u[idx2flat(i-1,j,Nx)] - u[idx]) / (dx^2)
        d2y = (u[idx2flat(i,j+1,Nx)] - 2u[idx] + u[idx2flat(i,j-1,Nx)]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif 2 <= i <= Nx - 1 && j == 1
        d2x = (u[idx2flat(i+1,j,Nx)] - 2u[idx] + u[idx2flat(i-1,j,Nx)]) / (dx^2)
        d2y = 2(u[idx2flat(i,j+1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif 2 <= i <= Nx - 1 && j == Ny
        d2x = (u[idx2flat(i+1,j,Nx)] - 2u[idx] + u[idx2flat(i-1,j,Nx)]) / (dx^2)
        d2y = 2(u[idx2flat(i,j-1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == 1 && j == 1
        d2x = 2(u[idx2flat(i+1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j+1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == Nx && j == 1
        d2x = 2(u[idx2flat(i-1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j+1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == 1 && j == Ny
        d2x = 2(u[idx2flat(i+1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j-1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == Nx && j == Ny
        d2x = 2(u[idx2flat(i-1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j-1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    end
end

"""Compute κ∇²u on a 2D regular grid into d2u."""
function diffusion_step!(d2u, u, t, κ, stim, dx, dy, Nx, Ny)
    backend = KA.get_backend(u)
    ∇²_2d_kernel!(backend)(d2u, u, t, κ, stim, dx, dy, Nx, Ny; ndrange=(Nx, Ny))
    KA.synchronize(backend)
    return nothing
end

# ============================================================
# 2. Cell update kernel (SatoBers forward Euler, GPU-compatible)
# ============================================================
"""
Cell forward Euler step: u_cells[:, i] += dt * cell_rhs(u_cells[:, i], x_i, t, model)
State layout: u_cells[state_index, node_index], state_index=1..15
"""
KA.@kernel function sato_cell_fwd_euler_kernel!(u_cells, t, dt, model)
    i = KA.@index(Global)   # node index 1..N_nodes

    # Compute x from node index
    ix = (i - 1) % model.nx          # 0-indexed x
    iy = (i - 1) ÷ model.nx          # 0-indexed y
    x  = Thunderbolt.Vec((Float64(ix) * model.dx, Float64(iy) * model.dy))

    # Load state into MVector (stack-allocated, no heap)
    u_local = MVector{15, Float64}(
        u_cells[1,i],  u_cells[2,i],  u_cells[3,i],  u_cells[4,i],  u_cells[5,i],
        u_cells[6,i],  u_cells[7,i],  u_cells[8,i],  u_cells[9,i],  u_cells[10,i],
        u_cells[11,i], u_cells[12,i], u_cells[13,i], u_cells[14,i], u_cells[15,i],
    )
    du_local = MVector{15, Float64}(0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0)

    # Compute ionic + stimulus RHS via SatoArmyHeart interface
    Thunderbolt.cell_rhs!(du_local, u_local, x, t, model)

    # Forward Euler update
    @inbounds for k in 1:15
        u_cells[k, i] = u_local[k] + dt * du_local[k]
    end
end

"""Run one cell-update step on all nodes in parallel."""
function cell_step!(u_cells, t::Float64, dt::Float64, model)
    N_nodes = size(u_cells, 2)
    backend = KA.get_backend(u_cells)
    sato_cell_fwd_euler_kernel!(backend)(u_cells, t, dt, model; ndrange=N_nodes)
    KA.synchronize(backend)
    return nothing
end

# ============================================================
# 3. Pseudo-ECG (Plonsey 1964, 2D)
#    Computes φ_e ∝ ∫∫ ∇V · (r - r_e) / |r - r_e|³ dA
# ============================================================
"""
    pseudo_ecg_2d(V_flat, Nx, Ny, dx, dy, xe, ye)

Compute pseudo-ECG at electrode (xe, ye) from voltage field on 2D regular grid.
V_flat is a flat (CPU) vector indexed as V[(j-1)*Nx + i] for node (i,j).
Returns scalar φ_e(t) in arbitrary units.
"""
function pseudo_ecg_2d(V_flat, Nx::Int, Ny::Int, dx::Float64, dy::Float64,
                        xe::Float64, ye::Float64)
    ecg = 0.0
    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = (j - 1) * Nx + i
            x_n = (i - 1) * dx
            y_n = (j - 1) * dy

            # Central-difference gradient (Neumann BCs: ghost nodes = same value)
            dVdx = if i == 1
                (V_flat[idx2flat(i+1,j,Nx)] - V_flat[idx]) / dx
            elseif i == Nx
                (V_flat[idx] - V_flat[idx2flat(i-1,j,Nx)]) / dx
            else
                (V_flat[idx2flat(i+1,j,Nx)] - V_flat[idx2flat(i-1,j,Nx)]) / (2dx)
            end

            dVdy = if j == 1
                (V_flat[idx2flat(i,j+1,Nx)] - V_flat[idx]) / dy
            elseif j == Ny
                (V_flat[idx] - V_flat[idx2flat(i,j-1,Nx)]) / dy
            else
                (V_flat[idx2flat(i,j+1,Nx)] - V_flat[idx2flat(i,j-1,Nx)]) / (2dy)
            end

            rx = x_n - xe;  ry = y_n - ye
            r  = sqrt(rx^2 + ry^2)
            r < 1e-8 && continue

            ecg += (dVdx * rx + dVdy * ry) / r^3 * dx * dy
        end
    end
    return ecg / (4π)
end

# ============================================================
# 4. QTVI computation
# ============================================================
"""
    compute_qtvi(ecg_times, ecg_signal, n_skip)

Compute QTVi from pseudo-ECG signal.
- Detect QRS onset (rapid upstroke) and T-wave end per beat
- QTVI = log10[SD²_QT / mean²_QT] (Berger et al. 1997)
"""
function compute_qtvi(ecg_times, ecg_signal; n_skip::Int=5, bcl::Float64=300.0)
    dt   = ecg_times[2] - ecg_times[1]
    npts = length(ecg_times)

    qt_intervals = Float64[]
    t = ecg_times[1]
    beat_idx = 0

    i = 1
    while i < npts - 50
        # Look for QRS: large negative deflection in ECG (R-wave corresponds to
        # activation front; for pseudo-ECG it's typically a biphasic signal)
        # We use a threshold crossing approach
        t_beat = ecg_times[i]
        beat_start_in_bcl = t_beat - floor(t_beat / bcl) * bcl

        # Detect R-wave: crossing from below → above threshold in rising phase
        # Look in first 30 ms of each beat window
        if beat_start_in_bcl < 5.0 && i > 1
            beat_idx += 1
            # Find T-wave end: last crossing of half-maximum after peak
            # within the same beat window
            t_end = t_beat + bcl - 20.0   # end of beat window (ms)
            j_end = searchsortedfirst(ecg_times, t_end)
            j_end = min(j_end, npts)

            # Find T-wave peak (max absolute value in the T-wave window: 50-280ms after QRS)
            t_twin_start = t_beat + 50.0
            t_twin_end   = t_beat + 280.0
            j_ts = searchsortedfirst(ecg_times, t_twin_start)
            j_te = min(searchsortedfirst(ecg_times, t_twin_end), npts)

            if j_ts >= j_te || j_ts > npts
                i += 1; continue
            end

            # Find T-wave peak in window
            t_peak_val = maximum(ecg_signal[j_ts:j_te])
            t_peak_neg = minimum(ecg_signal[j_ts:j_te])
            t_amp = abs(t_peak_val) > abs(t_peak_neg) ? t_peak_val : t_peak_neg
            t_peak_idx = findfirst(x -> x == t_amp, ecg_signal[j_ts:j_te])
            t_peak_idx === nothing && (i += 1; continue)
            t_peak_idx += j_ts - 1

            # T-wave end: 50% threshold crossing after peak
            threshold = 0.5 * t_amp
            t_end_idx = t_peak_idx
            for k in t_peak_idx:j_te
                if t_amp > 0
                    ecg_signal[k] <= threshold && (t_end_idx = k; break)
                else
                    ecg_signal[k] >= threshold && (t_end_idx = k; break)
                end
            end

            qt = ecg_times[t_end_idx] - t_beat
            qt > 50.0 && qt < 280.0 && beat_idx > n_skip && push!(qt_intervals, qt)
        end
        i += 1
    end

    length(qt_intervals) < 5 && return NaN

    mean_qt = mean(qt_intervals)
    var_qt  = var(qt_intervals)
    mean_qt < 10.0 && return NaN

    return log10(var_qt / mean_qt^2)
end

# ============================================================
# 5. Main simulation function
# ============================================================
"""
    run_2d_sim(; tauf, av, N_CaL, n_beats, ...)

Run a single 2D tissue simulation and return (ecg_times, ecg_signal, qtvi).
"""
function run_2d_sim(;
    tauf      :: Float64 = 52.0,
    av        :: Float64 = 3.0,
    N_CaL     :: Int     = 100_000,
    n_beats   :: Int     = 40,
    # Mesh
    L_MM      :: Float64 = 30.0,   # 3 cm
    N_EL      :: Int     = 60,
    # Physics
    kappa     :: Float64 = 0.1,    # mm²/ms diffusion coeff
    BCL       :: Float64 = 300.0,  # ms
    DT        :: Float64 = 0.05,   # ms
    DT_ECG    :: Float64 = 0.5,    # ms ECG sample interval
    stim_amp  :: Float64 = 80.0,
    stim_dur  :: Float64 = 1.0,
    corner_mm :: Float64 = 1.0,    # pacing corner size
    # Electrode
    xe        :: Float64 = L_MM + 5.0,
    ye        :: Float64 = L_MM + 5.0,
    use_gpu   :: Bool    = USE_GPU,
    verbose   :: Bool    = false,
)
    DX = L_MM / N_EL
    NX = N_EL + 1   # nodes in x
    NY = N_EL + 1   # nodes in y
    N_nodes = NX * NY

    # ---- Stability check ----
    cfl = kappa * DT / DX^2
    if cfl > 0.4
        @warn "CFL = $cfl > 0.4, may be unstable. Reduce DT or increase DX."
    end

    # ---- Build ionic model ----
    ionic = SatoBersArmyModel(;
        tauf=tauf, av=av, N_CaL=N_CaL,
        nx=NX, ny=NY, dx=DX, dy=DX, dt=DT, bcl=BCL,
        stim_amp=stim_amp, stim_dur=stim_dur, corner_mm=corner_mm,
    )

    # ---- Initial conditions (all nodes = resting state) ----
    s0 = Thunderbolt.default_initial_state(ionic)     # 15-element Vector{Float64}
    # u_cells[k, i] = state k of node i
    u_cells_cpu = Matrix{Float64}(undef, 15, N_nodes)
    for i in 1:N_nodes
        u_cells_cpu[:, i] .= s0
    end

    # ---- Move to GPU if requested ----
    if use_gpu
        u_cells = CuArray(u_cells_cpu)
    else
        u_cells = u_cells_cpu
    end

    # ---- Diffusion: no external stim (ionic model handles it) ----
    no_stim = (pos, t) -> 0.0

    # ---- Storage for ECG ----
    ECG_step = max(1, round(Int, DT_ECG / DT))  # sample every ECG_step timesteps
    n_steps = round(Int, BCL * n_beats / DT)
    ecg_capacity = div(n_steps, ECG_step) + 1
    ecg_times  = Vector{Float64}(undef, ecg_capacity)
    ecg_signal = Vector{Float64}(undef, ecg_capacity)
    ecg_idx = 0

    # Preallocate dv buffer
    if use_gpu
        dv = CUDA.zeros(Float64, N_nodes)
    else
        dv = zeros(Float64, N_nodes)
    end

    # ---- Time loop (Lie-Trotter: cell step → diffusion step) ----
    t = 0.0
    t_start = time()
    for step in 1:n_steps
        t = (step - 1) * DT

        # 1. Cell ODE step (includes stimulus from corner pacing)
        cell_step!(u_cells, t, DT, ionic)

        # 2. Diffusion step: v += DT * κ * ∇²v
        # Extract voltage from first row of u_cells
        v = @view u_cells[1, :]
        diffusion_step!(dv, v, t, kappa, no_stim, DX, DX, NX, NY)
        v .+= DT .* dv

        # 3. Sample ECG
        if mod(step, ECG_step) == 0
            ecg_idx += 1
            ecg_idx > ecg_capacity && break
            v_cpu = use_gpu ? Array(v) : Vector(v)
            ecg_times[ecg_idx]  = t
            ecg_signal[ecg_idx] = pseudo_ecg_2d(v_cpu, NX, NY, DX, DX, xe, ye)
        end

        if verbose && mod(step, round(Int, BCL / DT)) == 0
            beat = div(step, round(Int, BCL / DT))
            @printf("  beat %d/%d  t=%.1f ms  max(V)=%.1f mV  elapsed=%.1f s\n",
                beat, n_beats, t, use_gpu ? maximum(Array(v)) : maximum(v),
                time() - t_start)
        end
    end

    ecg_times  = ecg_times[1:ecg_idx]
    ecg_signal = ecg_signal[1:ecg_idx]

    qtvi = compute_qtvi(ecg_times, ecg_signal; n_skip=5, bcl=BCL)

    return ecg_times, ecg_signal, qtvi
end

# ============================================================
# 6. Parameter sweep and figure generation
# ============================================================
function run_tauf_sweep(tauf_vals; kwargs...)
    results = Dict{Float64, NamedTuple}()
    for (k, tauf) in enumerate(tauf_vals)
        @printf("τ_f sweep: %d/%d  tauf=%.1f\n", k, length(tauf_vals), tauf)
        t0 = time()
        times, ecg, qtvi = run_2d_sim(; tauf=tauf, kwargs...)
        dt = time() - t0
        @printf("  QTVi=%.3f  elapsed=%.1f s\n", qtvi, dt)
        results[tauf] = (times=times, ecg=ecg, qtvi=qtvi)
    end
    return results
end

function run_u_sweep(u_vals; kwargs...)
    results = Dict{Float64, NamedTuple}()
    for (k, u) in enumerate(u_vals)
        @printf("u sweep: %d/%d  u=%.2f\n", k, length(u_vals), u)
        t0 = time()
        times, ecg, qtvi = run_2d_sim(; av=u, kwargs...)
        dt = time() - t0
        @printf("  QTVi=%.3f  elapsed=%.1f s\n", qtvi, dt)
        results[u] = (times=times, ecg=ecg, qtvi=qtvi)
    end
    return results
end

function save_results(tauf_results, u_results; output_dir=joinpath(@__DIR__, "..", "data"))
    mkpath(output_dir)
    # τ_f sweep CSV
    tauf_file = joinpath(output_dir, "tissue2d_tauf_qtv.csv")
    open(tauf_file, "w") do f
        println(f, "tauf_ms,qtvi")
        for (tauf, r) in sort(collect(tauf_results))
            @printf(f, "%.2f,%.6f\n", tauf, r.qtvi)
        end
    end
    @info "Saved τ_f sweep → $tauf_file"

    # u sweep CSV
    u_file = joinpath(output_dir, "tissue2d_u_qtv.csv")
    open(u_file, "w") do f
        println(f, "u_inv_ms,qtvi")
        for (u, r) in sort(collect(u_results))
            @printf(f, "%.4f,%.6f\n", u, r.qtvi)
        end
    end
    @info "Saved u sweep → $u_file"

    return tauf_file, u_file
end

# ============================================================
# 7. Main entry point
# ============================================================
function main()
    println("="^60)
    println("2D Tissue FDM Simulation — Sato-Bers ArmyHeart Model")
    println("  GPU: $USE_GPU  |  Full sweep: $FULL_SWEEP")
    println("="^60)

    if !FULL_SWEEP
        # Quick test run
        println("\n[TEST] Single run: tauf=52.0, 10 beats, 20×20 mesh")
        t, ecg, qtvi = run_2d_sim(;
            tauf=52.0, n_beats=10, N_EL=20, L_MM=10.0,
            use_gpu=USE_GPU, verbose=true,
        )
        @printf("Test complete: %d ECG points, QTVi=%.4f\n", length(t), qtvi)
        println("Success ✓")
        return
    end

    # Full parameter sweeps
    println("\n=== τ_f sweep (Fig A2C) ===")
    tauf_vals = [20.0, 30.0, 35.0, 40.0, 45.0, 48.0, 51.0, 52.0, 55.0, 60.0]
    tauf_results = run_tauf_sweep(tauf_vals;
        n_beats=80, use_gpu=USE_GPU, verbose=true,
    )

    println("\n=== u sweep (Fig A2D) ===")
    u_vals = [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 13.0]
    u_results = run_u_sweep(u_vals;
        n_beats=80, use_gpu=USE_GPU, verbose=true,
    )

    save_results(tauf_results, u_results)
    println("\nAll done!")
end

main()
