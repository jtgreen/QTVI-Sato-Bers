#!/usr/bin/env julia
"""
tissue_armyheart.jl — 2D monodomain tissue simulation using ArmyHeart / Thunderbolt.jl

Reproduces Appendix Fig A2 from Sato et al. (2025):
  - 3 cm × 3 cm (60×60 quad elements → 61×61 nodes) 2D cardiac tissue
  - Corner pacing: stimulus at x < 1 mm, y < 1 mm (bottom-left)
  - Far-field pseudo-ECG via Thunderbolt's Plonsey1964ECGGaussCache
  - QT interval variability (QTVi) vs τ_f  (Fig A2C)
  - QT interval variability (QTVi) vs u     (Fig A2D)
  - Representative pseudo-ECG trace         (Fig A2B)

# ArmyHeart / Thunderbolt architecture used
  - generate_mesh (Ferrite.jl): regular 60×60 QuadrilateralQuad mesh
  - MonodomainModel (Thunderbolt): κ=0.1mm²/ms, χ=Cₘ=1
  - SatoBersArmyModel (SatoArmyHeart.jl): ionic + stochastic gating + stimulus
  - LieTrotterGodunov + BackwardEulerSolver + ForwardEulerCellSolver
  - Plonsey1964ECGGaussCache: far-field pseudo-ECG at electrode beyond the tissue
  - GPU: set use_gpu=true and ensure CUDA device is available

# GPU notes
  - Diffusion step uses BackwardEulerSolver with CUDA.CUSPARSE linear solver
  - Cell step uses ForwardEulerCellSolver with CuVector state arrays
  - Per-cell RNG states live in CuVector{UInt32} for thread-safe stochastic gating
  - Electrode position electrode: far corner opposite to pacing (≈32.5, 32.5 mm)
"""

using DrWatson
using Thunderbolt
import Thunderbolt: OrderedSet
using LinearSolve
using Statistics
using Printf

# Load SatoBersArmyModel
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================================
# GPU / CPU configuration
# ============================================================================
const USE_GPU = false   # Set true when a CUDA-capable GPU is available
                        # (and adjust device!() call below as needed)
if USE_GPU
    using CUDA
    device!(0)   # Use first GPU; change for multi-GPU (or use batch.jl)
end

const _precision = Float64

SolutionVectorType = USE_GPU ? CuVector{_precision} : Vector{_precision}
SystemMatrixType   = if USE_GPU
    CUDA.CUSPARSE.CuSparseMatrixCSR{_precision, Int32}
else
    Thunderbolt.ThreadedSparseMatrixCSR{_precision, Int32}
end

# ============================================================================
# Tissue geometry
# ============================================================================
const L      = 30.0          # Tissue length = width (mm)
const NEL    = 60            # Number of elements per side → 60×60 = 3600 cells
const DX     = L / NEL       # Element size = 0.5 mm
const NNX    = NEL + 1       # Nodes per side = 61
const NNY    = NEL + 1       # Nodes per side = 61

# Diffusion coefficient κ (mm²/ms), isotropic
# D = σᵢ/(χ Cₘ) = 0.1 mm²/ms → κ = D = 0.1
const KAPPA  = _precision(0.1)

# Electrode position: beyond the far corner opposite to the pacing site
# Paper: orange circle in Fig A2A  ≈ (32.5, 32.5 mm)
const X_ELEC = L + 2.5       # 32.5 mm (5 cells past far edge)
const Y_ELEC = L + 2.5       # 32.5 mm

# ============================================================================
# Simulation time parameters
# ============================================================================
const BCL      = 300.0        # Basic cycle length (ms)
const DT₀      = 0.05         # Time step (ms)
const STIM_AMP = 80.0         # µA/µF
const STIM_DUR = 1.0          # ms
const CORNER   = DX * 2       # Pacing region: x < 1 mm, y < 1 mm (2-element corner)

# ============================================================================
# Helper: build and run one tissue simulation
# Returns: (qt_intervals, ecg_trace, prepace_time_s, sim_time_s)
# ============================================================================
function run_tissue_sim(;
    tauf       :: Float64 = 52.0,
    av         :: Float64 = 3.0,
    N_CaL      :: Int     = 100000,
    gna        :: Float64 = 12.0,
    gkr        :: Float64 = 0.0136,
    gks        :: Float64 = 0.0245,
    g_rel      :: Float64 = 75.0,
    prepace_beats :: Int  = 20,
    meas_beats    :: Int  = 60,
    verbose       :: Bool = true,
)
    T   = _precision
    BCL_T = T(BCL)

    # ---- Build ionic model ----
    ionic = SatoBersArmyModel(;
        tauf, av, N_CaL, gna, gkr, gks, g_rel,
        bcl = BCL, stim_amp = STIM_AMP, stim_dur = STIM_DUR,
        corner_mm = CORNER,
        nx = NNX, ny = NNY, dx = DX, dy = DX,
        dt = DT₀,
    )

    # ---- Adapt to GPU if needed ----
    if USE_GPU
        ionic_gpu = Adapt.adapt(CUDA.cu, ionic)
    else
        ionic_gpu = ionic
    end

    # ---- Build mesh ----
    ep_mesh = generate_mesh(
        Quadrilateral,
        (NEL, NEL),
        Vec(T.((0.0, 0.0))),
        Vec(T.((L, L))),
    )
    ep_cs = CartesianCoordinateSystem(ep_mesh)

    # ---- MonodomainModel ----
    # χ = Cₘ = 1 (normalized), κ isotropic 2×2 tensor
    κ_tensor = ConstantCoefficient(SymmetricTensor{2, 2, T}((KAPPA, zero(T), KAPPA)))
    monodomain = MonodomainModel(
        ConstantCoefficient(one(T)),
        ConstantCoefficient(one(T)),
        κ_tensor,
        NoStimulationProtocol(),   # stimulus handled in cell_rhs!
        ionic_gpu,
        :φₘ,
        :s,
    )

    ip_ep = LagrangeCollection{1}()

    # ---- Semidiscretize ----
    odeform = semidiscretize(
        ReactionDiffusionSplit(monodomain, ep_cs),
        FiniteElementDiscretization(Dict(:φₘ => ip_ep)),
        ep_mesh,
    )

    # ---- Initial state ----
    u₀ = zeros(T, Thunderbolt.OS.function_size(odeform))
    Thunderbolt.steady_state_initializer!(u₀, odeform)

    # ---- Build ECG cache (Plonsey 1964) ----
    # We need the assembled diffusion operator for the ECG cache
    diff_op = Thunderbolt.setup_assembled_operator(
        Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice()),
        Thunderbolt.BilinearDiffusionIntegrator(κ_tensor, QuadratureRuleCollection(2), :φₘ),
        Thunderbolt.ThreadedSparseMatrixCSR{T, Int32},
        odeform.functions[1].dh,
    )
    Thunderbolt.update_operator!(diff_op, 0.0)

    # Build ECG cache with zero initial state
    φ₀_cpu = Vector{T}(u₀[odeform.solution_indices[1]])
    ecg_cache = Thunderbolt.Plonsey1964ECGGaussCache(diff_op, φ₀_cpu)
    electrode_vec = Vec(T.((X_ELEC, Y_ELEC)))
    κₜ = T(1.0)   # Equal conductivities approximation

    # ---- Solver setup (mirrors ArmyHeart 2D template) ----
    ep_stepper = BackwardEulerSolver(;
        inner_solver = LinearSolve.KrylovJL_CG(;
            itmax = 500,
            atol  = T(1e-8),
            rtol  = T(1e-8),
        ),
        solution_vector_type = SolutionVectorType,
        system_matrix_type   = SystemMatrixType,
    )

    cell_stepper = ForwardEulerCellSolver(;
        solution_vector_type = SolutionVectorType,
    )

    timestepper = Thunderbolt.OS.LieTrotterGodunov((ep_stepper, cell_stepper))

    # ---- Pre-pacing ----
    if verbose; @printf("  Pre-pacing %d beats (t=0..%.0f ms)...\n", prepace_beats, prepace_beats*BCL); end
    tpre = (zero(T), T(prepace_beats * BCL))
    prob_pre = Thunderbolt.OS.OperatorSplittingProblem(odeform, SolutionVectorType(u₀), tpre)
    integ_pre = Thunderbolt.OS.init(prob_pre, timestepper; dt=T(DT₀), verbose=false)
    t_pre_start = time()
    Thunderbolt.OS.solve!(integ_pre)
    t_pre_elapsed = time() - t_pre_start
    if verbose; @printf("  Pre-pacing done in %.1f s\n", t_pre_elapsed); end
    u_post_pre = Vector(integ_pre.u)

    # ---- Measurement phase ----
    if verbose; @printf("  Measurement: %d beats (t=0..%.0f ms)...\n", meas_beats, meas_beats*BCL); end
    tspan  = (zero(T), T(meas_beats * BCL))
    prob   = Thunderbolt.OS.OperatorSplittingProblem(odeform, SolutionVectorType(u_post_pre), tspan)
    integ  = Thunderbolt.OS.init(prob, timestepper; dt=T(DT₀), verbose=false)

    # ECG storage: sample every DT₀ ms
    total_steps = round(Int, meas_beats * BCL / DT₀)
    ecg_trace   = zeros(T, total_steps)
    dh          = odeform.functions[1].dh

    t_sim_start = time()
    step_idx = 0

    for (u_, t) in Thunderbolt.OS.TimeChoiceIterator(integ, range(tspan[1], tspan[2]; step=T(DT₀)))
        step_idx += 1
        if step_idx > total_steps; break; end

        # Extract φₘ (on CPU for ECG computation)
        φ_indices = odeform.solution_indices[1]
        φ_cpu = Vector{T}(u_[φ_indices])

        # Update ECG cache and evaluate at electrode
        Thunderbolt.update_ecg!(ecg_cache, φ_cpu)
        ecg_trace[step_idx] = Thunderbolt.evaluate_ecg(ecg_cache, electrode_vec, κₜ)
    end
    t_sim_elapsed = time() - t_sim_start
    if verbose; @printf("  Measurement done in %.1f s\n", t_sim_elapsed); end

    # ---- QT interval detection ----
    qt_intervals = detect_qt_intervals(ecg_trace, meas_beats; dt=DT₀)
    if verbose
        qt_mean = isempty(qt_intervals) ? NaN : mean(qt_intervals)
        qt_std  = isempty(qt_intervals) ? NaN : std(qt_intervals)
        qtvi    = compute_qtvi_paced(qt_intervals, BCL)
        @printf("  QT = %.2f ± %.3f ms  QTVi = %.4f  (n=%d beats)\n",
                qt_mean, qt_std, qtvi, length(qt_intervals))
    end

    return qt_intervals, ecg_trace, t_pre_elapsed, t_sim_elapsed
end

# ============================================================================
# QT interval detection (from 1D cable simulation, adapted for tissue ECG)
# ============================================================================
function detect_qt_intervals(ecg::Vector{T}, n_beats::Int; dt::Float64=DT₀) where T
    QTs   = T[]
    steps = round(Int, BCL / dt)
    for b in 0:n_beats-1
        i0 = b * steps + 1
        i1 = min((b+1)*steps, length(ecg))
        i1 <= i0 + 10 && continue
        beat = ecg[i0:i1]
        n    = length(beat)

        # Baseline from first + last 3%
        w = max(1, round(Int, 0.03*n))
        bl = 0.5 * (mean(beat[1:w]) + mean(beat[end-w+1:end]))

        # QRS: max deviation in first 40%
        qrs_end  = round(Int, 0.40*n)
        qrs_dev  = abs.(beat[1:qrs_end] .- bl)
        isempty(qrs_dev) && continue
        qrs_pidx = argmax(qrs_dev)
        qrs_sign = sign(beat[qrs_pidx] - bl)

        # Q onset: first crossing of 20% of QRS peak
        thresh_q  = bl + 0.20 * qrs_sign * qrs_dev[qrs_pidx]
        q_idx     = 1
        for i in 1:qrs_pidx
            if (qrs_sign > 0 ? beat[i] > thresh_q : beat[i] < thresh_q)
                q_idx = i; break
            end
        end

        # T-wave peak in 35%-95%
        t_s = round(Int, 0.35*n)
        t_e = round(Int, 0.95*n)
        t_s >= t_e && continue
        t_dev   = abs.(beat[t_s:t_e] .- bl)
        t_pidx  = argmax(t_dev) + t_s - 1
        t_sign  = sign(beat[t_pidx] - bl)
        t_dev_p = t_dev[argmax(t_dev)]

        # T-end: last crossing of 5% of T peak
        thresh_t  = bl + 0.05 * t_sign * t_dev_p
        t_end_idx = t_e
        for i in t_e:-1:t_pidx
            if (t_sign > 0 ? beat[i] > thresh_t : beat[i] < thresh_t)
                t_end_idx = i; break
            end
        end

        qt = (t_end_idx - q_idx) * dt
        if 50.0 < qt < 400.0
            push!(QTs, T(qt))
        end
    end
    return QTs
end

function compute_qtvi_paced(qts::Vector{T}, bcl::Float64) where T
    isempty(qts) && return T(NaN)
    length(qts) < 5 && return T(NaN)
    μ = mean(qts); σ² = var(qts)
    σ² = max(σ², T(1e-8))
    var_rr = T(1e-6)  # constant BCL → essentially zero RR variance
    return log10(σ²/μ^2) - log10(var_rr/bcl^2)
end

# ============================================================================
# Main: sweep τ_f and u, write CSV, save ECG traces
# ============================================================================
function main()
    println("="^72)
    println("2D Tissue Simulation — ArmyHeart / Thunderbolt.jl")
    println("Tissue: $(NEL)×$(NEL) quads, DX=$(DX)mm, κ=$(KAPPA)mm²/ms")
    println("Electrode: ($X_ELEC, $Y_ELEC) mm (far corner)")
    println("GPU: $USE_GPU")
    println("="^72)

    data_dir = joinpath(@__DIR__, "..", "figures", "data")
    mkpath(data_dir)

    prepace = 20   # beats — keep modest; paper uses 30 but 20 is sufficient for
    meas    = 60   # beats   preliminary results

    # ──────────────────────────────────────────────────────────────────────
    # τ_f sweep → QTVi vs τ_f  (Fig A2C)
    # ──────────────────────────────────────────────────────────────────────
    println("\n=== τ_f sweep → QTVi (Fig A2C) ===")
    tauf_vals = [20.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 52.0]

    ecg_saved  = Dict{String, Vector{Float64}}()

    open(joinpath(data_dir, "tissue2d_tauf_qtv.csv"), "w") do io
        println(io, "tauf,qt_mean,qt_std,qtvi,n_beats")
        for tf in tauf_vals
            @printf("\n  τ_f = %.1f ms\n", tf)
            qts, ecg, _, _ = run_tissue_sim(;
                tauf=tf, av=3.0, N_CaL=100000,
                prepace_beats=prepace, meas_beats=meas, verbose=true,
            )
            if length(qts) < 5
                println(io, "$tf,NaN,NaN,NaN,$(length(qts))")
            else
                μ_qt = mean(qts); σ_qt = std(qts)
                qtvi = compute_qtvi_paced(qts, BCL)
                println(io, "$tf,$μ_qt,$σ_qt,$qtvi,$(length(qts))")
                flush(io)
                # Save ECG for selected τ_f values
                if tf in [30.0, 45.0, 52.0]
                    ecg_saved["tauf$(round(Int,tf))"] = ecg
                end
            end
        end
    end
    println("  → wrote tissue2d_tauf_qtv.csv")

    # ──────────────────────────────────────────────────────────────────────
    # u sweep → QTVi vs u  (Fig A2D)
    # ──────────────────────────────────────────────────────────────────────
    println("\n=== u sweep → QTVi (Fig A2D) ===")
    u_vals = [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5]

    open(joinpath(data_dir, "tissue2d_u_qtv.csv"), "w") do io
        println(io, "u,qt_mean,qt_std,qtvi,n_beats")
        for uv in u_vals
            @printf("\n  u = %.2f ms⁻¹\n", uv)
            qts, ecg, _, _ = run_tissue_sim(;
                tauf=35.0, av=uv, N_CaL=100000,
                prepace_beats=prepace, meas_beats=meas, verbose=true,
            )
            if length(qts) < 5
                println(io, "$uv,NaN,NaN,NaN,$(length(qts))")
            else
                μ_qt = mean(qts); σ_qt = std(qts)
                qtvi = compute_qtvi_paced(qts, BCL)
                println(io, "$uv,$μ_qt,$σ_qt,$qtvi,$(length(qts))")
                flush(io)
                if uv in [4.0, 8.0, 9.5]
                    ecg_saved["u$(uv)"] = ecg
                end
            end
        end
    end
    println("  → wrote tissue2d_u_qtv.csv")

    # ──────────────────────────────────────────────────────────────────────
    # Save representative ECG traces
    # ──────────────────────────────────────────────────────────────────────
    if !isempty(ecg_saved)
        open(joinpath(data_dir, "tissue2d_ecg_traces.csv"), "w") do io
            keys_sorted = sort(collect(keys(ecg_saved)))
            println(io, "t_ms," * join(keys_sorted, ","))
            n = minimum(length(v) for v in values(ecg_saved))
            for i in 1:n
                t_ms = (i-1) * DT₀
                vals = join([string(ecg_saved[k][i]) for k in keys_sorted], ",")
                println(io, "$t_ms,$vals")
            end
        end
        println("  → wrote tissue2d_ecg_traces.csv")
    end

    println("\n=== 2D Tissue Simulation Complete ===")
    println("Run figures/plot_tissue2d_figures.py to generate plots.")
end

main()
