#!/usr/bin/env julia
"""
tissue_armyheart.jl — 2D monodomain tissue simulation using ArmyHeart / Thunderbolt.jl

Reproduces Appendix Fig A2 from Sato et al. (2025):
  - 3 cm × 3 cm, 60×60 Q1 elements (61×61 = 3721 nodes)
  - Corner pacing at x < 1 mm, y < 1 mm
  - Pseudo-ECG via Thunderbolt's Plonsey1964ECGGaussCache
  - QTVi vs τ_f (Fig A2C) and QTVi vs u (Fig A2D)
  - Stochastic ICaL gating (N_CaL=100,000) → beat-to-beat APD variability

# GPU path (9 × A30)
Set USE_GPU=true and call device!(gpu_id) before running.
Each GPU handles one parameter point; use ArmyHeart's batch.jl to
distribute the τ_f / u sweep across multiple GPUs.

# Diffusion solver
BackwardEuler with CUSPARSE CG on GPU — exploits the regular (orthogonal)
Cartesian mesh structure: the stiffness matrix is the standard discrete
Laplacian with constant coefficients, assembled once and reused every beat.
For the 61×61 grid this is a 3721×3721 5-banded matrix — very fast on A30.

# How to run
  julia tissue_armyheart.jl                 # CPU (default, USE_GPU=false)
  julia -e 'using CUDA; device!(0)' tissue_armyheart.jl   # GPU 0

# How to run the full τ_f sweep across multiple GPUs (using ArmyHeart batch.jl)
See the companion script: tissue_armyheart_batch.jl
"""

# Activate ArmyHeart environment if available (provides Thunderbolt, LinearSolve, etc.)
const ARMYHEART_PATH = get(ENV, "ARMYHEART_PATH", "/tmp/ArmyHeart")
if isdir(joinpath(ARMYHEART_PATH, "Project.toml"))
    import Pkg
    Pkg.activate(ARMYHEART_PATH; io=devnull)
    Pkg.instantiate(; io=devnull)
    println("Using ArmyHeart environment: $ARMYHEART_PATH")
else
    println("Using default Julia environment (Thunderbolt must be installed)")
end

using Thunderbolt
import Thunderbolt: OrderedSet
using LinearSolve
using Statistics
using Printf

# Load our Sato-Bers ArmyHeart model
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================================
# GPU / CPU toggle
# ============================================================================
const USE_GPU = parse(Bool, get(ENV, "SATO_USE_GPU", "false"))
if USE_GPU
    using CUDA
    @info "GPU mode: $(CUDA.device())"
end

const _T = Float64   # Use Float64 throughout (A30 has good FP64)

_vec_type() = USE_GPU ? CuVector{_T} : Vector{_T}
_mat_type() = USE_GPU ?
    CUDA.CUSPARSE.CuSparseMatrixCSR{_T, Int32} :
    Thunderbolt.ThreadedSparseMatrixCSR{_T, Int32}

# ============================================================================
# Tissue parameters  (match Sato et al. 2025, Appendix Fig A2)
# ============================================================================
const L      = 30.0      # Tissue side length (mm)
const N_EL   = 60        # Elements per side → 60×60 = 3600 Q1 elements
const DX     = L / N_EL  # = 0.5 mm node spacing
const N_NX   = N_EL + 1  # = 61 nodes in x
const N_NY   = N_EL + 1  # = 61 nodes in y

# κ = D_diff = 0.1 mm²/ms (isotropic),  χ = Cₘ = 1 (normalised)
const KAPPA  = _T(0.1)

# Electrode: 2.5 mm beyond the far corner opposite the pacing site
const X_ELEC = _T(L + 2.5)
const Y_ELEC = _T(L + 2.5)

# Pacing
const BCL      = 300.0   # ms
const DT₀      = 0.05    # ms  (explicit step for cell solver)
const STIM_AMP = 80.0    # µA/µF
const STIM_DUR = 1.0     # ms
const CORNER   = DX * 2  # pacing region x,y < 1 mm (2-element wide corner)

# ============================================================================
# Build one tissue simulation, run it, return QT intervals + ECG
# ============================================================================

function run_tissue_sim(;
    tauf          :: Float64 = 52.0,
    av            :: Float64 = 3.0,
    N_CaL         :: Int     = 100000,
    gna           :: Float64 = 12.0,
    gkr           :: Float64 = 0.0136,
    gks           :: Float64 = 0.0245,
    g_rel         :: Float64 = 75.0,
    prepace_beats :: Int     = 20,
    meas_beats    :: Int     = 60,
    verbose        :: Bool   = true,
)
    verbose && @printf("\n══ τ_f=%.1f  u=%.2f  N_CaL=%d ══\n", tauf, av, N_CaL)

    # ---- Ionic model ----
    ionic = SatoBersArmyModel(;
        tauf, av, N_CaL, gna, gkr, gks, g_rel,
        nx = N_NX, ny = N_NY, dx = DX, dy = DX, dt = DT₀,
        bcl = BCL, stim_amp = STIM_AMP, stim_dur = STIM_DUR, corner_mm = CORNER,
    )
    ionic_dev = USE_GPU ? Adapt.adapt(CUDA.cu, ionic) : ionic

    # ---- Mesh: 60×60 regular Q1 (Quadrilateral) grid ----
    # This is the "orthogonal geometry" — constant-coefficient Laplacian,
    # assembled once, very efficient on GPU (5-band structured matrix).
    ep_mesh = generate_mesh(
        Quadrilateral,
        (N_EL, N_EL),
        Vec(_T.((0.0, 0.0))),
        Vec(_T.((L, L))),
    )
    ep_cs = CartesianCoordinateSystem(ep_mesh)

    κ_coeff = ConstantCoefficient(SymmetricTensor{2, 2, _T}((KAPPA, zero(_T), KAPPA)))

    # ---- MonodomainModel ----
    monodomain = MonodomainModel(
        ConstantCoefficient(one(_T)),  # χ = 1
        ConstantCoefficient(one(_T)),  # Cₘ = 1
        κ_coeff,
        NoStimulationProtocol(),       # stimulus handled in cell_rhs!
        ionic_dev,
        :φₘ,
        :s,
    )

    # ---- Semidiscretize ----
    odeform = semidiscretize(
        ReactionDiffusionSplit(monodomain, ep_cs),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        ep_mesh,
    )

    # ---- Initial state ----
    u0_size = Thunderbolt.OS.function_size(odeform)
    u₀ = zeros(_T, u0_size)
    # Populate ionic state DOFs with default resting values
    heat_dofrange  = odeform.solution_indices[1]
    ionic_dofrange = odeform.solution_indices[2]
    ionic_model_inner = odeform.functions[2].ode   # extract ionic model from ODE function
    u0_cell = Thunderbolt.default_initial_state(ionic_model_inner, nothing)
    n_dofs  = length(heat_dofrange)
    n_states = Thunderbolt.num_states(ionic_model_inner)
    s0flat = @view u₀[ionic_dofrange]
    s0mat  = reshape(s0flat, (n_dofs, n_states))
    for k in 1:n_states
        s0mat[:, k] .= u0_cell[k]
    end

    # ---- Assembled diffusion operator (for ECG) ----
    strategy = Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice())
    diff_op  = Thunderbolt.setup_assembled_operator(
        strategy,
        Thunderbolt.BilinearDiffusionIntegrator(κ_coeff, QuadratureRuleCollection(2), :φₘ),
        Thunderbolt.ThreadedSparseMatrixCSR{_T, Int32},
        odeform.functions[1].dh,
    )
    Thunderbolt.update_operator!(diff_op, 0.0)

    # ECG cache (Plonsey 1964 far-field formula)
    φ₀_cpu = zeros(_T, length(heat_dofrange))
    ecg_cache = Thunderbolt.Plonsey1964ECGGaussCache(diff_op, φ₀_cpu)
    electrode  = Vec(_T.((X_ELEC, Y_ELEC)))
    κₜ         = one(_T)   # equal conductivities: σᵢ/(4π σₑ) ≈ 1/(4π)

    # ---- Solver: LieTrotterGodunov(BackwardEuler|diffusion, ForwardEuler|cells) ----
    ep_stepper = BackwardEulerSolver(;
        inner_solver        = LinearSolve.KrylovJL_CG(; itmax=1000, atol=_T(1e-8), rtol=_T(1e-8)),
        solution_vector_type = _vec_type(),
        system_matrix_type   = _mat_type(),
    )
    cell_stepper = ForwardEulerCellSolver(; solution_vector_type = _vec_type())
    timestepper  = Thunderbolt.OS.LieTrotterGodunov((ep_stepper, cell_stepper))

    # ---- Pre-pacing ----
    verbose && @printf("  Pre-pacing %d beats...\n", prepace_beats)
    t_pre = time()
    pre_tspan = (_T(0), _T(prepace_beats * BCL))
    pre_prob  = Thunderbolt.OS.OperatorSplittingProblem(odeform, _vec_type()(u₀), pre_tspan)
    pre_integ = Thunderbolt.OS.init(pre_prob, timestepper; dt=_T(DT₀), verbose=false)
    Thunderbolt.OS.solve!(pre_integ)
    u_prepaced = Vector{_T}(pre_integ.u)
    verbose && @printf("  Pre-pacing done in %.1f s\n", time() - t_pre)

    # ---- Measurement phase ----
    verbose && @printf("  Measuring %d beats...\n", meas_beats)
    t_meas = time()
    meas_tspan = (_T(0), _T(meas_beats * BCL))
    meas_prob  = Thunderbolt.OS.OperatorSplittingProblem(odeform, _vec_type()(u_prepaced), meas_tspan)
    meas_integ = Thunderbolt.OS.init(meas_prob, timestepper; dt=_T(DT₀), verbose=false)

    # Record ECG at each timestep
    n_steps   = round(Int, meas_beats * BCL / DT₀)
    ecg_trace = zeros(_T, n_steps)
    step_idx  = 0
    dh        = odeform.functions[1].dh

    for (u_, t_) in Thunderbolt.OS.TimeChoiceIterator(
            meas_integ, range(_T(0), _T(meas_beats * BCL); step=_T(DT₀)))
        step_idx += 1
        step_idx > n_steps && break
        φ_cpu = Vector{_T}(u_[heat_dofrange])
        Thunderbolt.update_ecg!(ecg_cache, φ_cpu)
        ecg_trace[step_idx] = Thunderbolt.evaluate_ecg(ecg_cache, electrode, κₜ)
    end
    verbose && @printf("  Measurement done in %.1f s\n", time() - t_meas)

    # ---- QT interval detection + QTVi ----
    qt_intervals = detect_qt_intervals(ecg_trace, meas_beats; dt=DT₀)
    if verbose
        n = length(qt_intervals)
        if n >= 5
            μ = mean(qt_intervals); σ = std(qt_intervals)
            qtvi = compute_qtvi(qt_intervals)
            @printf("  QT = %.2f ± %.3f ms  QTVi = %.4f  (n=%d)\n", μ, σ, qtvi, n)
        else
            @printf("  WARNING: only %d QT intervals detected\n", n)
        end
    end

    return qt_intervals, ecg_trace
end

# ============================================================================
# QT detection  (same algorithm as tissue_simulation.jl 1D cable)
# ============================================================================

function detect_qt_intervals(ecg::Vector{T}, n_beats::Int; dt::Float64=DT₀) where T
    QTs  = T[]
    step = round(Int, BCL / dt)
    for b in 0:n_beats-1
        i0 = b*step + 1; i1 = min((b+1)*step, length(ecg))
        i1 - i0 < 50 && continue
        beat = ecg[i0:i1]; n = length(beat)
        w  = max(1, round(Int, 0.03n))
        bl = 0.5*(mean(beat[1:w]) + mean(beat[end-w+1:end]))

        qe = round(Int, 0.40n)
        qd = abs.(beat[1:qe] .- bl)
        isempty(qd) && continue
        qp = argmax(qd); qs = sign(beat[qp] - bl)
        qt = bl + 0.20*qs*qd[qp]; qi = 1
        for i in 1:qp
            if (qs>0 ? beat[i]>qt : beat[i]<qt); qi=i; break; end
        end

        ts = round(Int, 0.35n); te0 = round(Int, 0.95n)
        ts >= te0 && continue
        td = abs.(beat[ts:te0] .- bl)
        tp = argmax(td) + ts - 1; tsgn = sign(beat[tp]-bl); tpk = td[argmax(td)]
        tt = bl + 0.05*tsgn*tpk; tei = te0
        for i in te0:-1:tp
            if (tsgn>0 ? beat[i]>tt : beat[i]<tt); tei=i; break; end
        end

        qt_ms = (tei - qi) * dt
        50.0 < qt_ms < 400.0 && push!(QTs, T(qt_ms))
    end
    return QTs
end

function compute_qtvi(qts::Vector)
    length(qts) < 5 && return NaN
    μ = mean(qts); σ² = max(var(qts), 1e-8)
    # Under constant BCL pacing, Var(RR) → 0; floor to avoid -Inf
    return log10(σ²/μ^2) - log10(1e-6/BCL^2)
end

# ============================================================================
# Main: τ_f sweep + u sweep → write CSV + save ECG traces
# ============================================================================

function main()
    println("="^70)
    println("2D Tissue — ArmyHeart/Thunderbolt.jl  ($(N_EL)×$(N_EL) quads)")
    println("GPU=$USE_GPU  κ=$(KAPPA)mm²/ms  electrode=($(X_ELEC),$(Y_ELEC))mm")
    println("="^70)

    out = joinpath(@__DIR__, "..", "figures", "data")
    mkpath(out)

    prepace = 20; meas = 60
    ecg_saved = Dict{String,Vector{Float64}}()

    # ── τ_f sweep ────────────────────────────────────────────────────────────
    println("\n=== τ_f sweep → QTVi (Fig A2C) ===")
    tauf_vals = [20.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 52.0]
    open(joinpath(out, "tissue2d_tauf_qtv.csv"), "w") do io
        println(io, "tauf,qt_mean,qt_std,qtvi,n_beats")
        for tf in tauf_vals
            qts, ecg = run_tissue_sim(; tauf=tf, av=3.0, N_CaL=100000,
                                        prepace_beats=prepace, meas_beats=meas)
            if length(qts) < 5
                println(io, "$tf,NaN,NaN,NaN,$(length(qts))")
            else
                μ=mean(qts); σ=std(qts); qtvi=compute_qtvi(qts)
                println(io, "$tf,$μ,$σ,$qtvi,$(length(qts))")
                flush(io)
            end
            tf in [30.0, 45.0, 52.0] && (ecg_saved["tauf$(round(Int,tf))"] = ecg)
        end
    end
    println("  → tissue2d_tauf_qtv.csv")

    # ── u sweep ──────────────────────────────────────────────────────────────
    println("\n=== u sweep → QTVi (Fig A2D) ===")
    u_vals = [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5]
    open(joinpath(out, "tissue2d_u_qtv.csv"), "w") do io
        println(io, "u,qt_mean,qt_std,qtvi,n_beats")
        for uv in u_vals
            qts, ecg = run_tissue_sim(; tauf=35.0, av=uv, N_CaL=100000,
                                        prepace_beats=prepace, meas_beats=meas)
            if length(qts) < 5
                println(io, "$uv,NaN,NaN,NaN,$(length(qts))")
            else
                μ=mean(qts); σ=std(qts); qtvi=compute_qtvi(qts)
                println(io, "$uv,$μ,$σ,$qtvi,$(length(qts))")
                flush(io)
            end
            uv in [4.0, 8.0, 9.5] && (ecg_saved["u$(uv)"] = ecg)
        end
    end
    println("  → tissue2d_u_qtv.csv")

    # ── Save ECG traces ───────────────────────────────────────────────────────
    if !isempty(ecg_saved)
        ks = sort(collect(keys(ecg_saved)))
        open(joinpath(out, "tissue2d_ecg_traces.csv"), "w") do io
            println(io, "t_ms," * join(ks, ","))
            n = minimum(length(v) for v in values(ecg_saved))
            for i in 1:n
                println(io, "$(round((i-1)*DT₀, digits=3))," * join([string(ecg_saved[k][i]) for k in ks], ","))
            end
        end
        println("  → tissue2d_ecg_traces.csv")
    end

    println("\n✓ 2D tissue simulation complete.")
end

main()
