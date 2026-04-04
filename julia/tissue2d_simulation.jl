#!/usr/bin/env julia
"""
tissue2d_simulation.jl — 2D monodomain tissue simulation of the Sato-Bers model

Reproduces Appendix Fig A2 from Sato et al. (2025):
  - 3 cm × 3 cm (60×60 cells at dx=0.5 mm) 2D cardiac tissue
  - Corner pacing (stimulus applied to bottom-left 2×2 corner)
  - Far-field pseudo-ECG at recording electrode
  - QT interval variability (QTV) vs τ_f  (Fig A2C)
  - QT interval variability (QTV) vs u     (Fig A2D)
  - Representative pseudo-ECG trace        (Fig A2B)

# Physics — 2D Monodomain Equation

  ∂V/∂t = D(∂²V/∂x² + ∂²V/∂y²) − Iion(V,s)/Cₘ + Istim

  D = σᵢ/(χCₘ) ≈ 0.1 mm²/ms  (isotropic approximation)
  dx = dy = 0.5 mm, dt = 0.05 ms
  Stability: D·dt/dx² = 0.02 ≪ 0.5 ✓

  Boundary conditions: no-flux (Neumann) at all edges.

# Pseudo-ECG (Plonsey 1964, 2D)

  φₑ(t) = α ∫∫ ∇V · ∇(1/r) dA
         ≈ α Σᵢⱼ [(∂V/∂x)ᵢⱼ (xᵢ−xₑ)/rᵢⱼ³
                 + (∂V/∂y)ᵢⱼ (yⱼ−yₑ)/rᵢⱼ³] · dx²

  α = σᵢ/(4π σₑ) ≈ 1/(4π) (equal conductivities approximation)
  Electrode placed beyond the far corner (opposite from stimulus).

# ArmyHeart integration note

  SatoBers.jl already implements the Thunderbolt.jl interface:
    cell_rhs!(du, u, model, stim)
    num_states(model) = 15
    transmembranepotential_index(model) = 1
    default_initial_state(model)

  When ArmyHeart.jl is accessible:
    using ArmyHeart
    tissue = ArmyHeart.Tissue2D(nx=NX, ny=NY, dx=DX, dy=DX,
                                cell=SatoBersModel(), D=D_DIFF)
    results = ArmyHeart.simulate!(tissue; bcl=300.0, beats=MEAS_BEATS,
                                  stim_region=CartesianIndices((1:2, 1:2)))
    ecg = ArmyHeart.pseudo_ecg_2d(results; electrode=(X_ELEC, Y_ELEC))
    qt_intervals = ArmyHeart.detect_qt(ecg)
    qtv = std(qt_intervals)
"""

include(joinpath(@__DIR__, "SatoBers.jl"))
using .SatoBers
using Printf
using Statistics

# ============================================================================
# Tissue geometry & simulation parameters
# ============================================================================
const NX     = 60           # Cells in x (30 mm / 0.5 mm spacing)
const NY     = 60           # Cells in y (30 mm / 0.5 mm spacing)
const DX     = 0.5          # Cell spacing (mm)
const D_DIFF = 0.1          # Diffusion coefficient [mm²/ms]
const DT     = 0.05         # Time step (ms)

const BCL      = 300.0      # Basic cycle length (ms)
const BCLn     = round(Int, BCL / DT)   # = 6000 steps/beat
const STIM_DUR = 1.0        # Stimulus duration (ms)
const STIM_AMP = 80.0       # Stimulus amplitude (µA/µF)
const STIM_n   = round(Int, STIM_DUR / DT)  # = 20 steps
const STIM_NX  = 2          # Stimulus region: (1:STIM_NX) × (1:STIM_NY) corner
const STIM_NY  = 2

# Electrode position (beyond far corner from stimulus)
# Paper: electrode near corner opposite to pacing site
const X_ELEC = (NX + 5) * DX  # 5 cells past the far edge  (32.5 mm)
const Y_ELEC = (NY + 5) * DX  # same  (32.5 mm)

# Pre-computed electrode weight factors for pseudo-ECG
# wx[i,j] = (x_i − x_e)/r³,  wy[i,j] = (y_j − y_e)/r³
const _wx = zeros(Float64, NX, NY)
const _wy = zeros(Float64, NX, NY)
begin
    for i in 1:NX, j in 1:NY
        xi = (i - 0.5) * DX
        yj = (j - 0.5) * DX
        r3 = ((xi - X_ELEC)^2 + (yj - Y_ELEC)^2)^1.5
        _wx[i, j] = (xi - X_ELEC) / r3
        _wy[i, j] = (yj - Y_ELEC) / r3
    end
end
const ECG_ALPHA = 1.0 / (4π)   # σᵢ/(4π σₑ) ≈ 1/(4π)
const ECG_dA    = DX^2          # Area element

# ============================================================================
# State arrays for all tissue cells
# State layout: U[i, j, k] where k=1..15 (matches SatoBers state vector)
#   k=1: V  (membrane potential)
#   k=2: ci, k=3: cs, k=4: cj, k=5: cjp, k=6: Ir
#   k=7: f,  k=8: q,  k=9: d
#   k=10: h, k=11: j
#   k=12: XKr, k=13: XKs, k=14: Xto, k=15: Yto
# ============================================================================

"""
    make_tissue_state(model) -> (U, RNG)

Initialise tissue state. U[NX, NY, 15], RNG[NX, NY] (xorshift32 per cell).
Each cell starts at the model's default initial state with slight noise.
"""
function make_tissue_state(model::SatoBersModel)
    u0  = collect(default_initial_state(model))
    U   = zeros(Float64, NX, NY, 15)
    for k in 1:15
        U[:, :, k] .= u0[k]
    end
    # Unique RNG seed per cell (XOR of base seed with cell index)
    RNG = [UInt32(1821800813) ⊻ UInt32(i * 65537 + j * 127) for i in 1:NX, j in 1:NY]
    return U, RNG
end

# ============================================================================
# 2D Laplacian (no-flux boundary conditions)
# ============================================================================
"""
Compute diffusion contribution dV[i,j] = D * ∇²V[i,j] with Neumann BCs.
Returns result in pre-allocated output array `dV`.
"""
function laplacian2d!(dV::Matrix{Float64}, V::Matrix{Float64})
    coeff = D_DIFF / DX^2
    @inbounds for j in 1:NY, i in 1:NX
        Vc = V[i, j]
        Vl = i > 1  ? V[i-1, j] : Vc   # Left  (or mirror)
        Vr = i < NX ? V[i+1, j] : Vc   # Right
        Vd = j > 1  ? V[i, j-1] : Vc   # Down
        Vu = j < NY ? V[i, j+1] : Vc   # Up
        dV[i, j] = coeff * (Vl + Vr + Vd + Vu - 4Vc)
    end
end

# ============================================================================
# Tissue time step
# ============================================================================
"""
    tissue_step!(U, RNG, dV_diff, u_buf, du_buf, model, tn)

Advance all NX×NY cells by one time step DT.
  - tn: global time step index (used to determine stimulus timing)
"""
function tissue_step!(U    ::Array{Float64, 3},
                      RNG  ::Matrix{UInt32},
                      dV_diff::Matrix{Float64},
                      u_buf::Vector{Float64},
                      du_buf::Vector{Float64},
                      model::SatoBersModel, tn::Int)
    beat_pos = mod(tn, BCLn)
    stim_on  = beat_pos < STIM_n

    # --- 1. Ion currents + intracellular state update ---
    @inbounds for j in 1:NY, i in 1:NX
        # Pack state vector
        for k in 1:15; u_buf[k] = U[i, j, k]; end

        # Stimulus only in corner region
        stim = (stim_on && i <= STIM_NX && j <= STIM_NY) ? STIM_AMP : 0.0

        # Compute deterministic RHS
        cell_rhs_deterministic!(du_buf, u_buf, model, stim)

        # Stochastic gate update (modifies u_buf[7,8,9] in-place and zeros du_buf[7,8,9])
        if model.N_CaL > 0
            rng = StochasticState(RNG[i, j])
            stochastic_gate_update!(u_buf, du_buf, DT, model, rng)
            RNG[i, j] = rng.xsx   # save updated PRNG state (field is .xsx, not .state)
        end

        # Forward Euler: update all state variables except V (k=1) and stochastic gates
        # Stochastic gates (k=7,8,9) were already updated in u_buf by stochastic_gate_update!
        # du_buf[7,8,9] were zeroed, so `U[i,j,k] += du_buf[k]*DT` is a no-op for them.
        for k in 2:15
            U[i, j, k] += du_buf[k] * DT
        end
        # Copy stochastic gate values directly from u_buf (bypasses du_buf)
        if model.N_CaL > 0
            U[i, j, 7] = u_buf[7]   # f gate
            U[i, j, 8] = u_buf[8]   # q gate
            U[i, j, 9] = u_buf[9]   # d gate
        end
        # Store ionic dV for later addition with diffusion
        dV_diff[i, j] = du_buf[1] * DT  # ionic contribution to ΔV
    end

    # --- 2. Add diffusion to V ---
    V_view = view(U, :, :, 1)        # NX×NY view of V
    V_mat  = V_view                   # alias
    lap    = similar(dV_diff)
    laplacian2d!(lap, V_mat)

    # --- 3. Update V = V + (ionic_dV + diffusion*DT) ---
    @. V_view += dV_diff + lap * DT
end

# ============================================================================
# Pseudo-ECG
# ============================================================================
"""
Compute pseudo-ECG scalar at current time from membrane potential matrix V.
"""
function pseudo_ecg(V::AbstractMatrix{Float64})
    ecg = 0.0
    @inbounds for j in 1:NY, i in 1:NX
        dVdx = (i < NX ? V[i+1,j] - V[i,j] : V[i,j] - V[i-1,j]) / DX
        dVdy = (j < NY ? V[i,j+1] - V[i,j] : V[i,j] - V[i,j-1]) / DX
        ecg += dVdx * _wx[i,j] + dVdy * _wy[i,j]
    end
    return ECG_ALPHA * ecg * ECG_dA
end

# ============================================================================
# QT interval detection from ECG trace
# ============================================================================
"""
    detect_qt_intervals(ecg, dt) -> Vector{Float64}

Given a multi-beat ECG trace (sampled at `dt` ms), detect QT interval in each
beat by:
  - Q-point: first upstroke of QRS (ECG crosses 20% of QRS peak value)
  - T-end: last time ECG falls below 10% of T-wave peak (after QRS)
"""
function detect_qt_intervals(ecg::Vector{Float64}, n_beats::Int; dt::Float64=DT)
    QTs = Float64[]
    steps_per_beat = BCLn

    for b in 0:n_beats-1
        idx0 = b * steps_per_beat + 1
        idx1 = min((b+1) * steps_per_beat, length(ecg))
        beat = ecg[idx0:idx1]
        n    = length(beat)
        n < 100 && continue

        # Baseline: mean of first and last 2%
        bl = 0.5 * (mean(beat[1:max(1,round(Int,0.02n))]) +
                    mean(beat[max(1,round(Int,0.98n)):n]))

        # QRS: find max |deviation| in first 40% of beat
        qrs_end = round(Int, 0.40 * n)
        qrs_devs = abs.(beat[1:qrs_end] .- bl)
        qrs_peak_idx = argmax(qrs_devs)
        qrs_peak = beat[qrs_peak_idx]
        qrs_sign = sign(qrs_peak - bl)

        # Q onset: first crossing of 20% of QRS peak
        q_thresh = bl + 0.20 * qrs_sign * qrs_devs[qrs_peak_idx]
        q_idx = 1
        for i in 1:qrs_peak_idx
            if qrs_sign > 0 ? beat[i] > q_thresh : beat[i] < q_thresh
                q_idx = i; break
            end
        end

        # T-wave: find max |deviation| in 35%-95% of beat
        t_start = round(Int, 0.35 * n)
        t_end0  = round(Int, 0.95 * n)
        t_devs  = abs.(beat[t_start:t_end0] .- bl)
        isempty(t_devs) && continue
        t_peak_local = argmax(t_devs)
        t_peak_abs   = t_start + t_peak_local - 1
        t_peak_sign  = sign(beat[t_peak_abs] - bl)
        t_peak_dev   = t_devs[t_peak_local]

        # T-end: last crossing of 5% of T-peak
        t_thresh = bl + 0.05 * t_peak_sign * t_peak_dev
        t_end_idx = t_end0
        for i in t_end0:-1:t_peak_abs
            if t_peak_sign > 0 ? beat[i] > t_thresh : beat[i] < t_thresh
                t_end_idx = i; break
            end
        end

        qt = (t_end_idx - q_idx) * dt
        if qt > 30.0 && qt < 400.0  # Physiological sanity check
            push!(QTs, qt)
        end
    end
    return QTs
end

# ============================================================================
# Full tissue simulation run
# ============================================================================
"""
    run_tissue(model; prepace, meas_beats) -> (QT_intervals, ecg_trace)

Run 2D tissue simulation. Returns QT intervals and ECG trace for last 5 beats.
"""
function run_tissue(model::SatoBersModel;
                    prepace::Int   = 30,
                    meas_beats::Int = 80,
                    verbose::Bool  = false)
    U, RNG = make_tissue_state(model)
    dV     = zeros(Float64, NX, NY)
    u_buf  = zeros(Float64, 15)
    du_buf = zeros(Float64, 15)

    # ECG recording (measurement phase only)
    ecg = zeros(Float64, meas_beats * BCLn)
    total_steps = (prepace + meas_beats) * BCLn
    meas_start  = prepace * BCLn

    for tn in 0:total_steps-1
        tissue_step!(U, RNG, dV, u_buf, du_buf, model, tn)
        if tn >= meas_start
            local_tn = tn - meas_start
            ecg[local_tn + 1] = pseudo_ecg(view(U, :, :, 1))
        end
        if verbose && mod(tn, 50000) == 0
            @printf("  Step %d / %d (beat %.1f)\n", tn, total_steps, tn/BCLn)
        end
    end

    QTs = detect_qt_intervals(ecg, meas_beats)
    return QTs, ecg
end

# ============================================================================
# Output directory
# ============================================================================
mkpath(joinpath(@__DIR__, "..", "figures", "data"))

println("=" ^ 70)
println("2D Tissue Simulation: $(NX)×$(NY) = $(NX*NY) cells, DX=$(DX)mm")
println("BCL=$(BCL)ms, DT=$(DT)ms → $(BCLn) steps/beat")
println("Paper: Sato et al. (2025) Appendix Fig A2")
println("=" ^ 70)

# ============================================================================
# τ_f sweep: QTV vs τ_f (Fig A2C)
# ============================================================================
println("\n=== τ_f sweep → QTV (Fig A2C) ===")

tauf_values = [20.0, 25.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 52.0]
PREPACE = 30   # Pre-pace beats (tissue reaches approximate steady state)
MEAS    = 80   # Measurement beats (more beats → better statistics)

ecg_saved = Dict{String, Vector{Float64}}()   # Key: "taufXX" or "uYY"

open(joinpath(@__DIR__, "..", "figures", "data", "tissue2d_tauf_qtv.csv"), "w") do io
    println(io, "tauf,qt_mean,qt_std,qtvi,n_beats")

    for tf in tauf_values
        t0 = time()
        model = SatoBersModel{Float64}(tauf=tf, av=3.0, N_CaL=100000)
        @printf("  τ_f=%5.1f ms: ", tf)
        flush(stdout)

        QTs, ecg = run_tissue(model; prepace=PREPACE, meas_beats=MEAS)

        if length(QTs) < 5
            println("  WARNING: only $(length(QTs)) QT intervals detected, skipping")
            println(io, "$tf,NaN,NaN,NaN,$(length(QTs))")
            continue
        end

        qt_mean = mean(QTs)
        qt_std  = std(QTs)
        qtvi    = log10(qt_std^2 / qt_mean^2)
        elapsed = round(time() - t0, digits=0)

        @printf("QT = %6.2f ± %5.3f ms  QTVi = %6.3f  (%d s, n=%d)\n",
                qt_mean, qt_std, qtvi, elapsed, length(QTs))
        println(io, "$tf,$qt_mean,$qt_std,$qtvi,$(length(QTs))")
        flush(io)

        # Save representative ECG traces
        if tf in [30.0, 45.0, 52.0]
            ecg_saved["tauf$(round(Int,tf))"] = ecg[end-5*BCLn+1:end]
        end
    end
end
println("  -> wrote figures/data/tissue2d_tauf_qtv.csv")

# ============================================================================
# u sweep: QTV vs u (Fig A2D)
# ============================================================================
println("\n=== u sweep → QTV (Fig A2D) ===")

u_values = [3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 9.5]

open(joinpath(@__DIR__, "..", "figures", "data", "tissue2d_u_qtv.csv"), "w") do io
    println(io, "u,qt_mean,qt_std,qtvi,n_beats")

    for u_val in u_values
        t0 = time()
        model = SatoBersModel{Float64}(tauf=35.0, av=u_val, N_CaL=100000)
        @printf("  u=%5.2f ms⁻¹: ", u_val)
        flush(stdout)

        QTs, ecg = run_tissue(model; prepace=PREPACE, meas_beats=MEAS)

        if length(QTs) < 5
            println("  WARNING: only $(length(QTs)) QT intervals detected, skipping")
            println(io, "$u_val,NaN,NaN,NaN,$(length(QTs))")
            continue
        end

        qt_mean = mean(QTs)
        qt_std  = std(QTs)
        qtvi    = log10(qt_std^2 / qt_mean^2)
        elapsed = round(time() - t0, digits=0)

        @printf("QT = %6.2f ± %5.3f ms  QTVi = %6.3f  (%d s, n=%d)\n",
                qt_mean, qt_std, qtvi, elapsed, length(QTs))
        println(io, "$u_val,$qt_mean,$qt_std,$qtvi,$(length(QTs))")
        flush(io)

        if u_val in [4.0, 8.0, 9.5]
            ecg_saved["u$(round(u_val,digits=1))"] = ecg[end-5*BCLn+1:end]
        end
    end
end
println("  -> wrote figures/data/tissue2d_u_qtv.csv")

# ============================================================================
# Save ECG traces for representative conditions
# ============================================================================
if !isempty(ecg_saved)
    println("\n=== Saving ECG traces ===")
    all_keys = sort(collect(keys(ecg_saved)))
    open(joinpath(@__DIR__, "..", "figures", "data", "tissue2d_ecg_traces.csv"), "w") do io
        println(io, "t_ms," * join(all_keys, ","))
        n_pts = minimum(length(v) for v in values(ecg_saved))
        for t_idx in 1:n_pts
            t_ms = (t_idx - 1) * DT
            vals = join([string(ecg_saved[k][t_idx]) for k in all_keys], ",")
            println(io, "$t_ms,$vals")
        end
    end
    println("  -> wrote figures/data/tissue2d_ecg_traces.csv")
end

# ============================================================================
# Single-beat activation map snapshot (fast preview)
# ============================================================================
println("\n=== Activation map snapshot (τ_f=40 ms) ===")
let
    model = SatoBersModel{Float64}(tauf=40.0, av=3.0, N_CaL=100000)
    U, RNG = make_tissue_state(model)
    dV     = zeros(Float64, NX, NY)
    u_buf  = zeros(Float64, 15)
    du_buf = zeros(Float64, 15)

    # Light prepace (5 beats)
    for b in 1:5, tn in 0:BCLn-1
        tissue_step!(U, RNG, dV, u_buf, du_buf, model, (b-1)*BCLn + tn)
    end

    # Record snapshots during one beat
    snap_ms = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 280.0]
    snap_tn = Set(round(Int, t/DT) for t in snap_ms)
    snap_V  = Dict(t => zeros(NX, NY) for t in snap_ms)

    for tn in 0:BCLn-1
        tissue_step!(U, RNG, dV, u_buf, du_buf, model, 5*BCLn + tn)
        if tn in snap_tn
            idx = findfirst(==(tn), round.(Int, snap_ms ./ DT))
            if idx !== nothing
                t_val = snap_ms[idx]
                snap_V[t_val] = copy(view(U, :, :, 1))
            end
        end
    end

    open(joinpath(@__DIR__, "..", "figures", "data", "tissue2d_activation.csv"), "w") do io
        col_names = join(["V_t$(round(Int,t))ms" for t in snap_ms], ",")
        println(io, "x_mm,y_mm,$col_names")
        for j in 1:NY, i in 1:NX
            x = (i - 0.5) * DX
            y = (j - 0.5) * DX
            vals = join([string(snap_V[t][i,j]) for t in snap_ms], ",")
            println(io, "$x,$y,$vals")
        end
    end
    println("  -> wrote figures/data/tissue2d_activation.csv")
end

println("\n=== 2D Tissue Simulation Complete ===")
println("Run figures/plot_tissue2d_figures.py to generate plots.")
