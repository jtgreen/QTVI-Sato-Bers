#!/usr/bin/env julia
"""
tissue_simulation.jl — 1D cable tissue simulation of the Sato-Bers model

Implements a monodomain 1D cable (finite difference) to study:
1. APD variability reduction with electrotonic coupling
2. Pseudo-ECG computation (Plonsey 1964 far-field formula)
3. Beat-to-beat QT interval variability (QTVI)
4. Behaviour near alternans bifurcation (tauf sweep)

# Physics

Monodomain cable equation (1D, Neumann BC):
    χ Cₘ ∂V/∂t = ∂/∂x(σᵢ ∂V/∂x) − χ Iion(V, s) + Istim
    ∂s/∂t = f(V, s)

Finite difference discretisation (forward Euler, dx = 0.5 mm, dt = 0.05 ms):
    dVᵢ/dt = D*(V_{i+1} − 2Vᵢ + V_{i-1})/dx² − Iion(Vᵢ, sᵢ)/Cₘ + Istim
    D = σᵢ/(χ Cₘ) [mm²/ms]

Stability check: D·dt/dx² ≪ 0.5 (satisfied with default parameters).

# Pseudo-ECG

Far-field extracellular potential at electrode (Plonsey 1964):
    φₑ(xₑ, yₑ, t) = (σᵢ/4π σₑ) ∫ (∂V/∂x)(x,t) · (x−xₑ) / r³ dx
where r = √((x−xₑ)² + yₑ²).

Discretised:
    φₑ(t) ≈ α · Σᵢ [(V_{i+1}−V_{i-1})/(2Δx)] · (xᵢ−xₑ)/rᵢ³ · Δx
    α = σᵢ/(4π σₑ)

# QTVI

QTVI = log₁₀[Var(QT)/Mean²(QT)] − log₁₀[Var(RR)/Mean²(RR)]

Under constant BCL pacing, Var(RR)≈0. We use the raw QT variability index:
    QTVi = log₁₀[SD²_QT / Mean²_QT]
This is equivalent to QTVI when RR is constant (as in this pacing study).

# ArmyHeart note

ArmyHeart.jl (DerangedIons/ArmyHeart, private) extends Thunderbolt.jl with:
  - Cable1D / Tissue2D types for geometry
  - ArmyHeart.simulate!() operator-splitting solver
  - ArmyHeart.pseudo_ecg() / ArmyHeart.pseudo_ecg_2d() ECG functions
  - ArmyHeart.qtvi() QTVI computation

This self-contained simulation replicates that functionality directly.
The SatoBers.jl module already implements the Thunderbolt.jl cell interface
(num_states, default_initial_state, transmembranepotential_index, cell_rhs!).
When ArmyHeart becomes available:
    using ArmyHeart
    include("SatoBers.jl"); using .SatoBers
    cable = ArmyHeart.Cable1D(length=60.0, dx=0.5, cell=SatoBersModel(), D=0.1)
    results = ArmyHeart.simulate!(cable; bcl=300.0, beats=200)
    ecg = ArmyHeart.pseudo_ecg(results; electrode=(70.0, 5.0))
    println("QTVI = ", ArmyHeart.qtvi(ecg))

# Output

Writes CSV files to ../figures/data/:
  - tissue_apd_variability.csv  : APD std vs tauf: single cell vs tissue
  - tissue_qtvi.csv             : QTVI vs tauf
  - tissue_cable_snapshot.csv   : Voltage along cable at several times
  - tissue_ecg.csv              : Pseudo-ECG trace at representative tauf

References:
  Sato D et al. J Physiol (2025). DOI: 10.1113/JP289051
  Plonsey R. Biophysics (1964). Far-field ECG formula.
  Heijman J et al. Circ Res (2014). Tissue coupling reduces APD variability.
"""

include(joinpath(@__DIR__, "SatoBers.jl"))
using .SatoBers
using Printf

# ============================================================================
# Tissue parameters
# ============================================================================
const N_CELLS   = 100         # Number of cells in cable
const DX        = 0.5         # Cell spacing (mm)
const D_DIFF    = 0.1         # Diffusion coefficient D = σᵢ/(χCₘ) [mm²/ms]
                               # Typical: σᵢ ≈ 0.174 S/m, χ ≈ 1000/mm, Cₘ ≈ 1 µF/cm²
const DT        = 0.05        # Time step (ms) — same as single-cell
const BCL       = 300.0       # Basic cycle length (ms)
const BCLn      = round(Int, BCL / DT)
const STIM_DUR  = 1.0         # Stimulus duration (ms)
const STIM_AMPl = 80.0        # Stimulus amplitude (µA/µF)
const STIM_n    = round(Int, STIM_DUR / DT)

# Electrode position for pseudo-ECG
# Placed 10 mm beyond cable end, 5 mm to the side
const X_ELEC    = N_CELLS * DX + 10.0   # mm (beyond the end)
const Y_ELEC    = 5.0                    # mm (off-axis)

# Pseudo-ECG scaling: α = σᵢ/(4π σₑ)
# σᵢ ≈ σₑ in typical tissue → α ≈ 1/(4π) ≈ 0.0796
const ECG_ALPHA = 1.0 / (4π)

# Cell coordinates along cable (mm)
const X_CELLS = [(i - 1) * DX for i in 1:N_CELLS]

# Stability check
const COURANT = D_DIFF * DT / DX^2
# COURANT ≈ 0.02 → well within stability limit of 0.5

# ============================================================================
# Tissue state: V[1:N], s[1:N, 1:14]  (v = state 1, rest are states 2:15)
# ============================================================================
mutable struct TissueState
    V  :: Vector{Float64}      # Transmembrane potential at each node [mV]
    s  :: Matrix{Float64}      # Internal state variables [N × 14]
    du :: Matrix{Float64}      # Derivative buffer [N × 15] (reused each step)
    rng:: Vector{StochasticState}  # Per-cell PRNG for stochastic gating
end

"""
    init_tissue(model) -> TissueState

Initialise tissue at uniform resting state.
Each cell starts with the default initial conditions from the Sato-Bers model.
"""
function init_tissue(model::SatoBersModel{T}) where T
    u0 = collect(default_initial_state(model))   # 15-element vector
    V  = fill(u0[1], N_CELLS)
    s  = repeat(u0[2:end]', N_CELLS)   # [N_CELLS × 14]
    du = zeros(N_CELLS, 15)
    # Each cell gets its own PRNG, seeded uniquely to avoid correlations
    base_seed = UInt32(1821800813)
    rng = [StochasticState(base_seed + UInt32(i * 7919)) for i in 1:N_CELLS]
    return TissueState(V, s, du, rng)
end

# ============================================================================
# Single time step
# ============================================================================

"""
    tissue_step!(ts, model, stim_mask)

Advance tissue state by one time step dt.
`stim_mask[i]` is the stimulus current [µA/µF] applied to cell i.
Uses Forward Euler for the reaction and diffusion terms.
"""
function tissue_step!(ts::TissueState, model::SatoBersModel, stim_mask::Vector{Float64})
    N = N_CELLS
    V = ts.V
    s = ts.s

    # ------ Diffusion term: D * d²V/dx² (central FD, Neumann BC) ------
    dV_diff = similar(V)
    # Interior nodes: standard central difference
    @inbounds for i in 2:(N-1)
        dV_diff[i] = D_DIFF * (V[i+1] - 2*V[i] + V[i-1]) / DX^2
    end
    # Boundary nodes: no-flux (Neumann): V[0] = V[1], V[N+1] = V[N]
    dV_diff[1] = D_DIFF * (V[2] - V[1]) / DX^2
    dV_diff[N] = D_DIFF * (V[N-1] - V[N]) / DX^2

    # ------ Reaction term: cell ODE at each node ------
    u_local  = zeros(15)
    du_local = zeros(15)
    @inbounds for i in 1:N
        # Pack local state
        u_local[1] = V[i]
        u_local[2:end] .= @view s[i, :]

        # Compute deterministic RHS (fills du_local)
        cell_rhs_deterministic!(du_local, u_local, model, stim_mask[i])

        # Apply Langevin noise to ICaL gates if N_CaL > 0
        stochastic_gate_update!(u_local, du_local, DT, model, ts.rng[i])

        # Add diffusion to voltage derivative
        du_local[1] += dV_diff[i]

        # Forward Euler update for all state variables
        V[i]     = u_local[1] + DT * du_local[1]
        for k in 2:15
            s[i, k-1] = u_local[k] + DT * du_local[k]
        end
    end
    return nothing
end

# ============================================================================
# Pseudo-ECG computation
# ============================================================================

"""
    compute_pseudoecg(V, x_cells, x_elec, y_elec) -> Float64

Compute the pseudo-ECG signal at electrode position (x_elec, y_elec) using
the Plonsey (1964) far-field formula:
    φₑ ≈ α · Σᵢ (∂V/∂x)|ᵢ · (xᵢ − xₑ) / rᵢ³ · Δx
where rᵢ = √((xᵢ − xₑ)² + yₑ²).
"""
function compute_pseudoecg(V::Vector{Float64}, x_cells::Vector{Float64},
                           x_elec::Float64, y_elec::Float64)
    N = length(V)
    φ = 0.0
    # Central finite difference for dV/dx (with forward/backward at boundaries)
    @inbounds for i in 1:N
        xi  = x_cells[i]
        ri  = sqrt((xi - x_elec)^2 + y_elec^2)
        ri3 = ri^3

        # Compute ∂V/∂x at cell i
        if i == 1
            dVdx = (V[2] - V[1]) / DX
        elseif i == N
            dVdx = (V[N] - V[N-1]) / DX
        else
            dVdx = (V[i+1] - V[i-1]) / (2*DX)
        end

        φ += ECG_ALPHA * dVdx * (xi - x_elec) / ri3 * DX
    end
    return φ
end

# ============================================================================
# APD measurement (single cell, from tissue)
# ============================================================================

"""
    measure_apd(V_trace, dt; vc=-70.0) -> (apd, upstroke_t)

Measure APD from a voltage time trace at threshold vc [mV].
Uses linear interpolation for sub-dt accuracy.
"""
function measure_apd(V_trace::Vector{Float64}, dt::Float64; vc::Float64 = -70.0)
    t1 = NaN; t2 = NaN
    for i in 2:length(V_trace)
        vold = V_trace[i-1]; vnew = V_trace[i]
        t    = (i-2) * dt
        if isnan(t1) && vold < vc && vnew > vc
            t1 = t + dt * (vc - vold) / (vnew - vold)
        elseif !isnan(t1) && isnan(t2) && vold > vc && vnew < vc
            t2 = t + dt * (vc - vold) / (vnew - vold)
        end
    end
    if isnan(t1) || isnan(t2) || t2 < t1
        return NaN, NaN
    end
    return t2 - t1, t1
end

"""
    measure_qt_from_ecg(ecg, dt; threshold=0.0) -> Float64

Measure QT interval from a pseudo-ECG beat.
Q-wave onset: first crossing of ECG above threshold (upstroke of QRS).
T-wave end: last crossing of ECG back through zero after the T-wave peak.
"""
function measure_qt_from_ecg(ecg::Vector{Float64}, dt::Float64; threshold::Float64 = 0.0)
    n = length(ecg)
    # Find QRS onset (first positive excursion)
    t_q = NaN
    t_t = NaN
    for i in 2:n
        if isnan(t_q) && ecg[i-1] < threshold && ecg[i] > threshold
            t_q = (i-2) * dt + dt * (threshold - ecg[i-1]) / (ecg[i] - ecg[i-1])
        end
    end
    if isnan(t_q)
        return NaN
    end
    # Find T-wave end: last crossing of zero after peak
    for i in n:-1:2
        if ecg[i-1] > threshold && ecg[i] < threshold
            t_t = (i-2) * dt + dt * (threshold - ecg[i-1]) / (ecg[i] - ecg[i-1])
            break
        end
    end
    if isnan(t_t) || t_t < t_q
        return NaN
    end
    return t_t - t_q
end

# ============================================================================
# QTVI computation
# ============================================================================

"""
    compute_qtvi(qt_series, rr_series) -> Float64

Compute QTVI from series of QT intervals and RR intervals:
    QTVI = log₁₀[Var(QT)/Mean²(QT)] − log₁₀[Var(RR)/Mean²(RR)]

With constant BCL pacing, Var(RR) → 0; we add a small floor to avoid -Inf.
"""
function compute_qtvi(qt_series::Vector{Float64}, rr_series::Vector{Float64})
    valid = .!isnan.(qt_series) .& .!isnan.(rr_series)
    qt = qt_series[valid]; rr = rr_series[valid]
    length(qt) < 10 && return NaN

    var_qt  = var(qt);  mean_qt = mean(qt)
    var_rr  = var(rr);  mean_rr = mean(rr)

    # Floor to avoid log(0) — 0.01² ms² lower limit
    var_qt  = max(var_qt,  1e-4)
    var_rr  = max(var_rr,  1e-4)

    return log10(var_qt / mean_qt^2) - log10(var_rr / mean_rr^2)
end

# Convenience: QTVI under constant pacing (use just QT variability)
function compute_qtvi_paced(qt_series::Vector{Float64}, bcl::Float64)
    valid = .!isnan.(qt_series)
    qt = qt_series[valid]
    length(qt) < 10 && return NaN
    var_qt  = var(qt)
    mean_qt = mean(qt)
    var_rr  = 1e-6          # BCL is constant → essentially zero variability
    mean_rr = bcl
    var_qt  = max(var_qt, 1e-6)
    return log10(var_qt / mean_qt^2) - log10(var_rr / mean_rr^2)
end

# ============================================================================
# Simulation helper: run tissue for N beats, return APD + ECG per beat
# ============================================================================

"""
    run_tissue_beats(model, beats; prepace_beats=50) -> (apd_tissue, qt_ecg, ecg_trace)

Simulate tissue for `beats` measurement beats (after prepacing).
Returns:
  - apd_tissue[beat]: APD at the middle cell (electrotonic averaging)
  - qt_ecg[beat]: QT interval measured from pseudo-ECG
  - ecg_trace: the ECG for the last beat (for visualization)
  - cable_snapshots: V along cable at selected times (last beat)
"""
function run_tissue_beats(model::SatoBersModel{Float64}, beats::Int;
                          prepace_beats::Int = 50)
    ts = init_tissue(model)

    N     = N_CELLS
    mid   = div(N, 2)   # Middle cell for APD measurement

    stim_mask = zeros(N)   # Stimulus applied only to first 5 cells

    # ECG storage for one beat
    ecg_one_beat = Vector{Float64}(undef, BCLn)
    V_mid_beat   = Vector{Float64}(undef, BCLn)

    # ---------- Pre-pacing ----------
    println("  Pre-pacing $(prepace_beats) beats...")
    for beat in 1:prepace_beats
        for tn in 0:(BCLn-1)
            if tn < STIM_n
                stim_mask[1:5] .= STIM_AMPl
            else
                fill!(stim_mask, 0.0)
            end
            tissue_step!(ts, model, stim_mask)
        end
        if beat % 10 == 0
            print("\r  Prepace beat $beat/$prepace_beats    ")
            flush(stdout)
        end
    end
    println()

    # ---------- Measurement beats ----------
    println("  Measuring $(beats) beats...")
    apd_tissue   = Vector{Float64}(undef, beats)
    qt_ecg       = Vector{Float64}(undef, beats)
    last_ecg     = zeros(BCLn)
    last_V_cable = zeros(N_CELLS, 10)   # Snapshots at 10 time points in last beat

    for beat in 1:beats
        # Record voltage at mid-cell and ECG for this beat
        for tn in 0:(BCLn-1)
            if tn < STIM_n
                stim_mask[1:5] .= STIM_AMPl
            else
                fill!(stim_mask, 0.0)
            end
            tissue_step!(ts, model, stim_mask)
            ecg_one_beat[tn+1] = compute_pseudoecg(ts.V, X_CELLS, X_ELEC, Y_ELEC)
            V_mid_beat[tn+1]   = ts.V[mid]
        end

        # Measure APD at mid-cell
        apd_val, _ = measure_apd(V_mid_beat, DT)
        apd_tissue[beat] = apd_val

        # Measure QT from ECG
        qt_val = measure_qt_from_ecg(ecg_one_beat, DT)
        qt_ecg[beat] = qt_val

        if beat == beats
            last_ecg .= ecg_one_beat
            # Snapshots at 10 evenly spaced times
            for (k, tn) in enumerate(round.(Int, range(0, BCLn-1, length=10)))
                last_V_cable[:, k] .= ts.V
            end
        end

        if beat % 20 == 0
            print("\r  Beat $beat/$beats    ")
            flush(stdout)
        end
    end
    println()

    return apd_tissue, qt_ecg, last_ecg, last_V_cable
end

# ============================================================================
# Single-cell reference simulation (same tauf, no coupling)
# ============================================================================

"""
    run_singlecell_beats(model, beats; prepace_beats=100) -> (apd_sc, ci_sc)

Simulate a single uncoupled cell for comparison with tissue results.
"""
function run_singlecell_beats(model::SatoBersModel{Float64}, beats::Int;
                               prepace_beats::Int = 100)
    u  = collect(default_initial_state(model))
    du = zeros(15)
    rng = StochasticState(UInt32(1821800813))

    # Pre-pacing
    for _ in 1:prepace_beats
        for tn in 0:(BCLn-1)
            st = tn < STIM_n ? STIM_AMPl : 0.0
            cell_rhs_deterministic!(du, u, model, st)
            stochastic_gate_update!(u, du, DT, model, rng)
            @. u += du * DT
        end
    end

    # Measurement
    apd_sc = Vector{Float64}(undef, beats)
    ci_sc  = Vector{Float64}(undef, beats)
    V_beat = Vector{Float64}(undef, BCLn)

    for beat in 1:beats
        cimax = 0.0
        for tn in 0:(BCLn-1)
            st = tn < STIM_n ? STIM_AMPl : 0.0
            cell_rhs_deterministic!(du, u, model, st)
            stochastic_gate_update!(u, du, DT, model, rng)
            @. u += du * DT
            V_beat[tn+1] = u[1]
            cimax = max(cimax, u[2])
        end
        apd_val, _ = measure_apd(V_beat, DT)
        apd_sc[beat] = apd_val
        ci_sc[beat]  = cimax
    end

    return apd_sc, ci_sc
end

# ============================================================================
# Statistics helpers
# ============================================================================
function mean(x)
    filter_x = filter(!isnan, x)
    isempty(filter_x) && return NaN
    sum(filter_x) / length(filter_x)
end

function var(x)
    filter_x = filter(!isnan, x)
    length(filter_x) < 2 && return NaN
    m = sum(filter_x) / length(filter_x)
    sum((xi - m)^2 for xi in filter_x) / (length(filter_x) - 1)
end

function std(x)
    s = var(x)
    isnan(s) ? NaN : sqrt(s)
end

# ============================================================================
# Main: tauf sweep  —  single cell vs tissue comparison
# ============================================================================

function main()
    println("="^70)
    println("Sato-Bers 1D Cable Tissue Simulation")
    println("Cable: N=$(N_CELLS) cells, dx=$(DX) mm, D=$(D_DIFF) mm²/ms")
    println("Courant number: $(round(COURANT, digits=4)) (stable: < 0.5)")
    println("Electrode: ($(X_ELEC), $(Y_ELEC)) mm")
    println("="^70)

    # Output directory
    data_dir = joinpath(@__DIR__, "..", "figures", "data")
    mkpath(data_dir)

    # -----------------------------------------------------------------------
    # Scenario 1: tauf sweep — APD variability: single cell vs tissue
    # -----------------------------------------------------------------------
    # Sweep a few representative tauf values near/away from bifurcation
    tauf_vals = [30.0, 40.0, 45.0, 50.0, 52.0, 55.0, 60.0]
    BEATS_MEAS  = 100   # beats to measure (enough for variability estimate)
    BEATS_PRE   = 60    # prepace beats for tissue

    # Storage
    sc_apd_mean   = Float64[]
    sc_apd_std    = Float64[]
    tis_apd_mean  = Float64[]
    tis_apd_std   = Float64[]
    qtvi_vals     = Float64[]
    qt_mean_vals  = Float64[]
    qt_std_vals   = Float64[]

    for tf in tauf_vals
        println("\n--- tauf = $(tf) ms ---")
        model = SatoBersModel{Float64}(tauf=tf, av=3.0, N_CaL=100000)

        # Single-cell reference
        println("  [single cell]")
        apd_sc, _ = run_singlecell_beats(model, BEATS_MEAS; prepace_beats=100)

        # Tissue
        println("  [tissue]")
        apd_tis, qt_tis, _, _ = run_tissue_beats(model, BEATS_MEAS;
                                                  prepace_beats=BEATS_PRE)

        rr_series = fill(BCL, length(qt_tis))
        qtvi = compute_qtvi_paced(qt_tis, BCL)

        push!(sc_apd_mean,   mean(apd_sc))
        push!(sc_apd_std,    std(apd_sc))
        push!(tis_apd_mean,  mean(apd_tis))
        push!(tis_apd_std,   std(apd_tis))
        push!(qtvi_vals,     qtvi)
        push!(qt_mean_vals,  mean(qt_tis))
        push!(qt_std_vals,   std(qt_tis))

        @printf("  SC APD: %.1f ± %.2f ms  |  Tissue APD: %.1f ± %.2f ms  |  QTVI: %.3f\n",
                mean(apd_sc), std(apd_sc), mean(apd_tis), std(apd_tis), qtvi)
    end

    # Write APD variability CSV
    open(joinpath(data_dir, "tissue_apd_variability.csv"), "w") do io
        println(io, "tauf,sc_apd_mean,sc_apd_std,tis_apd_mean,tis_apd_std,qt_mean,qt_std,qtvi")
        for i in 1:length(tauf_vals)
            @printf(io, "%.1f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f,%.4f\n",
                    tauf_vals[i],
                    sc_apd_mean[i], sc_apd_std[i],
                    tis_apd_mean[i], tis_apd_std[i],
                    qt_mean_vals[i], qt_std_vals[i],
                    qtvi_vals[i])
        end
    end
    println("\nWrote tissue_apd_variability.csv")

    # -----------------------------------------------------------------------
    # Scenario 2: ECG trace at representative tauf (stable) and (near bifurcation)
    # -----------------------------------------------------------------------
    println("\n--- ECG traces ---")
    ecg_beats = 30
    ecg_pre   = 40

    for tf in [30.0, 52.0]
        model = SatoBersModel{Float64}(tauf=tf, av=3.0, N_CaL=100000)
        println("  tauf=$(tf): running $(ecg_beats) beats for ECG...")
        _, qt_tis, last_ecg, _ = run_tissue_beats(model, ecg_beats;
                                                    prepace_beats=ecg_pre)

        # Save last beat ECG
        fname = joinpath(data_dir, "tissue_ecg_tauf$(round(Int,tf)).csv")
        open(fname, "w") do io
            println(io, "t_ms,ecg_mV")
            for k in 1:BCLn
                @printf(io, "%.3f,%.6f\n", (k-1)*DT, last_ecg[k])
            end
        end
        println("  Wrote $(fname)")
    end

    # -----------------------------------------------------------------------
    # Scenario 3: Cable snapshot — action potential propagation (tauf=40 ms)
    # -----------------------------------------------------------------------
    println("\n--- Cable propagation snapshot ---")
    model_snap = SatoBersModel{Float64}(tauf=40.0, av=3.0, N_CaL=0)   # deterministic
    ts_snap = init_tissue(model_snap)
    stim_mask = zeros(N_CELLS)

    # Record V[i] at several times during one beat (after 20 prepace beats)
    SNAP_BEATS = 20
    for beat in 1:SNAP_BEATS
        for tn in 0:(BCLn-1)
            stim_mask[1:5] .= tn < STIM_n ? STIM_AMPl : 0.0
            tissue_step!(ts_snap, model_snap, stim_mask)
        end
    end

    # Now record one beat at selected time points
    snap_times_ms = [2.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 280.0]
    snap_tidx = round.(Int, snap_times_ms ./ DT)
    snap_data = Dict(t => zeros(N_CELLS) for t in snap_times_ms)

    for tn in 0:(BCLn-1)
        stim_mask[1:5] .= tn < STIM_n ? STIM_AMPl : 0.0
        tissue_step!(ts_snap, model_snap, stim_mask)
        for (t, tidx) in zip(snap_times_ms, snap_tidx)
            if tn == tidx
                snap_data[t] .= ts_snap.V
            end
        end
    end

    open(joinpath(data_dir, "tissue_cable_snapshot.csv"), "w") do io
        header = "x_mm," * join(["V_t$(round(Int,t))ms" for t in snap_times_ms], ",")
        println(io, header)
        for i in 1:N_CELLS
            row = @sprintf("%.2f", X_CELLS[i]) * "," *
                  join([@sprintf("%.4f", snap_data[t][i]) for t in snap_times_ms], ",")
            println(io, row)
        end
    end
    println("Wrote tissue_cable_snapshot.csv")

    println("\n" * "="^70)
    println("All tissue simulation data saved to $(data_dir)/")
    println("Run figures/plot_figures.py to generate plots.")
    println("="^70)
end

main()
