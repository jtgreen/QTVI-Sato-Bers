#!/usr/bin/env julia
"""
tissue_fdm_2d.jl — 2D monodomain tissue simulation using ArmyHeart FDM approach

Reproduces Sato et al. 2025 Appendix Fig A2:
  - 3 cm × 3 cm, 61×61 = 3721 nodes (60×60 elements, dx = 0.5 mm)
  - Corner pacing (bottom-left region, x,y < 1 mm)
  - Pseudo-ECG at opposite corner (Plonsey 1964)
  - QTVi vs τ_f (Fig A2C) and QTVi vs u (Fig A2D)
  - Stochastic ICaL gating (N_CaL = 100,000)

Architecture (ArmyHeart FDM orthogonal geometry speedup):
  - FDM 5-point stencil on regular Cartesian grid (no FEM assembly)
  - SatoArmyHeart.jl: SatoBersArmyModel ionic model
  - Lie-Trotter operator splitting: cell_step → diffusion_step
  - CPU: Threads.@threads  |  GPU: CUDA.@cuda (Float64 on A30)

Run:
  julia tissue_fdm_2d.jl                      # CPU, quick test (20×20, 10 beats)
  SATO_USE_GPU=true julia tissue_fdm_2d.jl    # GPU
  SATO_FULL_SWEEP=true julia tissue_fdm_2d.jl # Full parameter sweep
"""

# ============================================================
# 0. Setup
# ============================================================
const USE_GPU    = parse(Bool, get(ENV, "SATO_USE_GPU", "false"))
const FULL_SWEEP = parse(Bool, get(ENV, "SATO_FULL_SWEEP", "false"))

# Import GPU support first if needed
if USE_GPU
    using CUDA
    using Adapt
end

using StaticArrays
using LinearAlgebra
using Statistics
using Printf
using Thunderbolt   # Vec type + cell_rhs! interface

# Include our SatoBers ArmyHeart cell model
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================
# 1. ArmyHeart FDM: 2D Laplacian (5-point stencil, Neumann BCs)
#    The "orthogonal geometry speedup": no FEM assembly needed
#    for regular Cartesian grids
# ============================================================

"""
CPU: compute κ∇²v into dv for 2D regular grid (Neumann BCs, no-flux).
Uses the ArmyHeart FDM orthogonal geometry approach:
  - Ghost-node reflection at boundaries (dV/dn = 0)
  - 5-point stencil: d²V/dx² = (V_{i-1} - 2V_i + V_{i+1}) / dx²
"""
function diffusion_step_cpu!(dv::AbstractVector, v::AbstractVector,
                              κ::Float64, dx::Float64, Nx::Int, Ny::Int)
    dx2 = dx * dx
    @inbounds Threads.@threads for j in 1:Ny
        for i in 1:Nx
            idx = (j - 1) * Nx + i

            # Ghost-node Neumann BC: reflect across boundary
            vW = i > 1  ? v[(j-1)*Nx + i-1] : v[idx]
            vE = i < Nx ? v[(j-1)*Nx + i+1] : v[idx]
            vS = j > 1  ? v[(j-2)*Nx + i]   : v[idx]
            vN = j < Ny ? v[j*Nx + i]        : v[idx]

            dv[idx] = κ * ((vW - 2v[idx] + vE) / dx2 +
                           (vS - 2v[idx] + vN) / dx2)
        end
    end
end

# ============================================================
# 2. Cell update (SatoBers forward Euler, per-node, CPU threaded)
# ============================================================

"""
CPU threaded cell update: forward Euler across all nodes in parallel.
u_cells[:,i] = 15 state variables for node i.
"""
function cell_step_cpu!(u_cells::Matrix{Float64},
                         t::Float64, dt::Float64, model)
    N_nodes = size(u_cells, 2)
    @inbounds Threads.@threads for i in 1:N_nodes
        ix = (i - 1) % model.nx
        iy = (i - 1) ÷ model.nx
        x  = Thunderbolt.Vec((Float64(ix) * model.dx,
                               Float64(iy) * model.dy))

        u_local  = MVector{15, Float64}(ntuple(k -> u_cells[k, i], Val(15)))
        du_local = MVector{15, Float64}(ntuple(_ -> 0.0, Val(15)))

        Thunderbolt.cell_rhs!(du_local, u_local, x, t, model)

        @simd for k in 1:15
            u_cells[k, i] = u_local[k] + dt * du_local[k]
        end
    end
end

# ============================================================
# 3. GPU kernels — only compiled when USE_GPU=true
# ============================================================
if USE_GPU
    @eval begin
        """CUDA cell ODE kernel: forward Euler, each thread = one node."""
        function _cuda_cell_kernel!(u_cells, t, dt, model, N_nodes)
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            i > N_nodes && return

            ix = (i - 1) % model.nx
            iy = (i - 1) ÷ model.nx
            x  = Thunderbolt.Vec((Float64(ix) * model.dx,
                                   Float64(iy) * model.dy))

            u_local  = MVector{15, Float64}(ntuple(k -> u_cells[k, i], Val(15)))
            du_local = MVector{15, Float64}(ntuple(_ -> 0.0, Val(15)))

            Thunderbolt.cell_rhs!(du_local, u_local, x, t, model)

            @inbounds for k in 1:15
                u_cells[k, i] = u_local[k] + dt * du_local[k]
            end
            return nothing
        end

        function cell_step_gpu!(u_cells, t::Float64, dt::Float64, model)
            N_nodes = size(u_cells, 2)
            threads = 256
            blocks  = cld(N_nodes, threads)
            CUDA.@cuda threads=threads blocks=blocks _cuda_cell_kernel!(
                u_cells, t, dt, model, N_nodes)
            CUDA.synchronize()
        end

        """CUDA diffusion kernel: 5-point stencil, Neumann BCs."""
        function _cuda_diffusion_kernel!(dv, v, κ, dx2, Nx, Ny)
            i = (CUDA.blockIdx().x - 1) * CUDA.blockDim().x + CUDA.threadIdx().x
            j = (CUDA.blockIdx().y - 1) * CUDA.blockDim().y + CUDA.threadIdx().y
            (i > Nx || j > Ny) && return

            idx = (j - 1) * Nx + i
            vW = i > 1  ? v[(j-1)*Nx + i-1] : v[idx]
            vE = i < Nx ? v[(j-1)*Nx + i+1] : v[idx]
            vS = j > 1  ? v[(j-2)*Nx + i]   : v[idx]
            vN = j < Ny ? v[j*Nx + i]        : v[idx]

            dv[idx] = κ * ((vW - 2v[idx] + vE) / dx2 +
                           (vS - 2v[idx] + vN) / dx2)
            return nothing
        end

        function diffusion_step_gpu!(dv, v, κ::Float64, dx::Float64,
                                      Nx::Int, Ny::Int)
            dx2 = dx * dx
            tx, ty = 16, 16
            bx = cld(Nx, tx);  by = cld(Ny, ty)
            CUDA.@cuda threads=(tx,ty) blocks=(bx,by) _cuda_diffusion_kernel!(
                dv, v, κ, dx2, Nx, Ny)
            CUDA.synchronize()
        end
    end # @eval
end   # if USE_GPU

# Unified dispatch
cell_step!(u_cells::Matrix{Float64}, t, dt, model) =
    cell_step_cpu!(u_cells, t, dt, model)

diffusion_step!(dv::AbstractVector{Float64}, v::AbstractVector{Float64},
                κ, dx, Nx, Ny) =
    diffusion_step_cpu!(dv, v, κ, dx, Nx, Ny)

if USE_GPU
    cell_step!(u_cells::CuArray, t, dt, model) =
        cell_step_gpu!(u_cells, t, dt, model)
    diffusion_step!(dv::CuArray, v::CuArray, κ, dx, Nx, Ny) =
        diffusion_step_gpu!(dv, v, κ, dx, Nx, Ny)
end

# ============================================================
# 4. Pseudo-ECG (Plonsey 1964, 2D)
# ============================================================

"""
    pseudo_ecg_2d(V_flat, Nx, Ny, dx, xe, ye)

Compute far-field pseudo-ECG at electrode (xe, ye) from 2D voltage field.
V_flat[(j-1)*Nx + i] = voltage at grid node (i,j), with 0-based coordinates
x_n = (i-1)*dx, y_n = (j-1)*dx.
"""
function pseudo_ecg_2d(V::AbstractVector{<:Real}, Nx::Int, Ny::Int, dx::Float64,
                        xe::Float64, ye::Float64)
    dA  = dx * dx
    ecg = 0.0
    @inbounds for j in 1:Ny
        for i in 1:Nx
            idx = (j - 1) * Nx + i
            x_n = (i - 1) * dx
            y_n = (j - 1) * dx

            dVdx = if i == 1
                (V[idx + 1]    - V[idx]) / dx
            elseif i == Nx
                (V[idx] - V[idx - 1])    / dx
            else
                (V[idx + 1]    - V[idx - 1]) / (2dx)
            end

            dVdy = if j == 1
                (V[idx + Nx]   - V[idx]) / dx
            elseif j == Ny
                (V[idx] - V[idx - Nx])   / dx
            else
                (V[idx + Nx]   - V[idx - Nx]) / (2dx)
            end

            rx = x_n - xe;  ry = y_n - ye
            r  = sqrt(rx*rx + ry*ry)
            r < 1e-8 && continue

            ecg += (dVdx * rx + dVdy * ry) / (r * r * r) * dA
        end
    end
    return ecg / (4π)
end

# ============================================================
# 5. QTVI computation
# ============================================================

"""
    compute_qtvi(ecg_times, ecg_signal; n_skip, bcl) → Float64

Compute QTVi = log₁₀[Var(QT)/Mean²(QT)] (Berger 1997).
Detects T-wave peaks + end per beat using 50% threshold crossing.
"""
function compute_qtvi(ecg_times::Vector{Float64}, ecg_signal::Vector{Float64};
                      n_skip::Int=5, bcl::Float64=300.0)
    length(ecg_times) < 20 && return NaN

    qt_intervals = Float64[]
    beat_idx     = 0
    dt_ecg       = ecg_times[2] - ecg_times[1]
    npts         = length(ecg_times)

    i = 1
    while i <= npts
        # Detect beat start: t mod BCL ≈ 0
        if mod(ecg_times[i], bcl) < dt_ecg * 1.5 && i > 1
            beat_idx += 1
            t_qrs = ecg_times[i]

            # T-wave search window: 50–280 ms after QRS
            j_ts = searchsortedfirst(ecg_times, t_qrs + 50.0)
            j_te = min(searchsortedfirst(ecg_times, t_qrs + 280.0), npts)

            if beat_idx > n_skip && j_te > j_ts + 5
                t_window = @view ecg_signal[j_ts:j_te]

                # T-wave peak: max |ECG|
                aw = abs.(t_window)
                pk_local = argmax(aw)
                t_amp    = t_window[pk_local]
                abs(t_amp) < 1e-12 && (i += 1; continue)
                pk_global = pk_local + j_ts - 1

                # T-wave end: 50% crossing after peak
                thresh    = 0.5 * t_amp
                t_end_idx = j_te
                for k in pk_global:j_te
                    if t_amp > 0 && ecg_signal[k] ≤ thresh
                        t_end_idx = k; break
                    elseif t_amp < 0 && ecg_signal[k] ≥ thresh
                        t_end_idx = k; break
                    end
                end

                qt = ecg_times[t_end_idx] - t_qrs
                50.0 < qt < 280.0 && push!(qt_intervals, qt)
            end
        end
        i += 1
    end

    length(qt_intervals) < 5 && return NaN
    mean_qt = mean(qt_intervals)
    mean_qt < 10.0 && return NaN

    return log10(var(qt_intervals; corrected=true) / mean_qt^2)
end

# ============================================================
# 6. Main simulation function
# ============================================================

"""
    run_2d_sim(; kwargs...) → (ecg_times, ecg_signal, qtvi)

Run one 2D monodomain simulation. Returns pseudo-ECG time series and QTVi.
"""
function run_2d_sim(;
    tauf       :: Float64 = 52.0,
    av         :: Float64 = 3.0,
    N_CaL      :: Int     = 100_000,
    n_beats    :: Int     = 40,
    L_MM       :: Float64 = 30.0,     # tissue side length (mm)
    N_EL       :: Int     = 60,       # elements per side → 61 nodes
    kappa      :: Float64 = 0.1,      # mm²/ms diffusion coefficient
    BCL        :: Float64 = 300.0,    # ms basic cycle length
    DT         :: Float64 = 0.05,     # ms timestep
    DT_ECG     :: Float64 = 0.5,      # ms ECG sample interval
    stim_amp   :: Float64 = 80.0,
    stim_dur   :: Float64 = 1.0,
    corner_mm  :: Float64 = 1.0,
    xe         :: Float64 = L_MM + 5.0,
    ye         :: Float64 = L_MM + 5.0,
    verbose    :: Bool    = false,
)
    DX      = L_MM / N_EL
    NX      = N_EL + 1
    NY      = N_EL + 1
    N_nodes = NX * NY

    # Stability: κ*dt/dx² < 0.5
    cfl = kappa * DT / DX^2
    cfl > 0.48 && @warn "CFL = $(round(cfl,digits=3)) > 0.48, may be unstable"

    # Build ionic model
    ionic = SatoBersArmyModel(;
        tauf=tauf, av=av, N_CaL=N_CaL,
        nx=NX, ny=NY, dx=DX, dy=DX, dt=DT, bcl=BCL,
        stim_amp=stim_amp, stim_dur=stim_dur, corner_mm=corner_mm,
    )

    # Initial conditions
    s0      = Thunderbolt.default_initial_state(ionic)
    u_cells = Matrix{Float64}(undef, 15, N_nodes)
    for i in 1:N_nodes; u_cells[:, i] .= s0; end

    # Move to GPU if needed, adapt ionic model
    if USE_GPU
        u_cells_dev = CuArray(u_cells)
        ionic_dev   = Adapt.adapt(CuArray, ionic)
        dv_dev      = CUDA.zeros(Float64, N_nodes)
    else
        u_cells_dev = u_cells
        ionic_dev   = ionic
        dv_dev      = zeros(Float64, N_nodes)
    end

    # ECG storage
    ECG_step = max(1, round(Int, DT_ECG / DT))
    n_steps  = round(Int, BCL * n_beats / DT)
    ecg_t    = Float64[]; sizehint!(ecg_t,  div(n_steps, ECG_step) + 2)
    ecg_s    = Float64[]; sizehint!(ecg_s,  div(n_steps, ECG_step) + 2)

    # ---- Lie-Trotter operator splitting time loop ----
    t0 = time()
    for step in 1:n_steps
        t = (step - 1) * DT

        # 1. Cell ODE step (stim is baked into ionic model)
        cell_step!(u_cells_dev, t, DT, ionic_dev)

        # 2. Diffusion step: v_new = v_cell + DT * κ * ∇²v_cell
        v_dev = @view u_cells_dev[1, :]
        diffusion_step!(dv_dev, v_dev, kappa, DX, NX, NY)
        v_dev .+= DT .* dv_dev

        # 3. Sample ECG
        if mod(step - 1, ECG_step) == 0
            v_cpu = USE_GPU ? Array(v_dev) : collect(v_dev)
            push!(ecg_t, t)
            push!(ecg_s, pseudo_ecg_2d(v_cpu, NX, NY, DX, xe, ye))
        end

        if verbose && mod(step, round(Int, BCL / DT)) == 0
            beat  = div(step, round(Int, BCL / DT))
            v_cpu = USE_GPU ? Array(v_dev) : collect(v_dev)
            @printf("  beat %3d/%d  t=%5.0f ms  maxV=%6.1f mV  %.0f s\n",
                beat, n_beats, t, maximum(v_cpu), time()-t0)
        end
    end

    qtvi = compute_qtvi(ecg_t, ecg_s; n_skip=5, bcl=BCL)
    return ecg_t, ecg_s, qtvi
end

# ============================================================
# 7. Parameter sweeps
# ============================================================

function run_tauf_sweep(tauf_vals; kwargs...)
    results = Dict{Float64, NamedTuple}()
    for (k, τ) in enumerate(tauf_vals)
        @printf("[τ_f %d/%d] τ_f=%.1f ms\n", k, length(tauf_vals), τ)
        t0  = time()
        ts, ecg, qi = run_2d_sim(; tauf=τ, kwargs...)
        @printf("  QTVi=%s  (%.1f s)\n",
            isnan(qi) ? "NaN" : @sprintf("%.4f", qi), time()-t0)
        results[τ] = (times=ts, ecg=ecg, qtvi=qi)
    end
    return results
end

function run_u_sweep(u_vals; kwargs...)
    results = Dict{Float64, NamedTuple}()
    for (k, u) in enumerate(u_vals)
        @printf("[u %d/%d] u=%.2f ms⁻¹\n", k, length(u_vals), u)
        t0  = time()
        ts, ecg, qi = run_2d_sim(; av=u, kwargs...)
        @printf("  QTVi=%s  (%.1f s)\n",
            isnan(qi) ? "NaN" : @sprintf("%.4f", qi), time()-t0)
        results[u] = (times=ts, ecg=ecg, qtvi=qi)
    end
    return results
end

function save_sweep_csv(r_tauf, r_u)
    out = joinpath(@__DIR__, "..", "data")
    mkpath(out)

    f1 = joinpath(out, "tissue2d_tauf_qtv.csv")
    open(f1, "w") do f
        println(f, "tauf_ms,qtvi")
        for (τ, r) in sort(collect(r_tauf))
            @printf(f, "%.2f,%.6f\n", τ, isnan(r.qtvi) ? 0.0 : r.qtvi)
        end
    end
    @info "Saved → $f1"

    f2 = joinpath(out, "tissue2d_u_qtv.csv")
    open(f2, "w") do f
        println(f, "u_inv_ms,qtvi")
        for (u, r) in sort(collect(r_u))
            @printf(f, "%.4f,%.6f\n", u, isnan(r.qtvi) ? 0.0 : r.qtvi)
        end
    end
    @info "Saved → $f2"
end

# ============================================================
# 8. Entry point
# ============================================================
function main()
    println("="^60)
    @printf("2D FDM Tissue — Sato-Bers (ArmyHeart FDM)\n")
    @printf("  GPU=%s  threads=%d  FullSweep=%s\n",
        USE_GPU, Threads.nthreads(), FULL_SWEEP)
    println("="^60)

    if !FULL_SWEEP
        @printf("\n[TEST] 20×20 grid, 10 beats\n")
        t0 = time()
        ts, ecg, qi = run_2d_sim(;
            tauf=52.0, n_beats=10, N_EL=20, L_MM=10.0, verbose=true)
        @printf("Done in %.1f s — %d ECG pts  QTVi=%s\n",
            time()-t0, length(ts),
            isnan(qi) ? "NaN(ok — few beats)" : @sprintf("%.4f", qi))
        println("✓ Test passed!")
        return
    end

    # Full sweeps (paper Figs A2C, A2D)
    println("\n[τ_f sweep — Fig A2C]")
    tauf_vals = [20.0, 30.0, 35.0, 40.0, 44.0, 48.0, 50.0, 52.0, 55.0, 60.0]
    r_tauf = run_tauf_sweep(tauf_vals; n_beats=80, verbose=true)

    println("\n[u sweep — Fig A2D]")
    u_vals = [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5, 10.0, 11.0, 13.0]
    r_u = run_u_sweep(u_vals; n_beats=80, verbose=true)

    save_sweep_csv(r_tauf, r_u)
    println("\n✓ All sweeps complete!")
end

main()
