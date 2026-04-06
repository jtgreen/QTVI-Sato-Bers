#!/usr/bin/env julia
"""
tissue_armyheart.jl — 2D monodomain tissue simulation using ArmyHeart / Thunderbolt.jl

Reproduces Appendix Fig A2 from Sato et al. (2025):
  - 3 cm × 3 cm, 60×60 Q1 Quadrilateral elements (61×61 = 3721 nodes)
  - Corner pacing at x < 1 mm, y < 1 mm (baked into cell_rhs! of SatoBersArmyModel)
  - Pseudo-ECG via Thunderbolt's Plonsey1964ECGGaussCache (Plonsey 1964 far-field)
  - QTVi vs τ_f (Fig A2C) and QTVi vs u (Fig A2D)
  - Stochastic ICaL gating (N_CaL=100,000) → beat-to-beat APD variability

Architecture follows ArmyHeart's with_hetero_standalone.jl exactly:
  - MonodomainModel + NoStimulationProtocol (stimulus inside cell_rhs!)
  - OS.LieTrotterGodunov(BackwardEulerSolver, AdaptiveForwardEulerSubstepper)
  - GPU: CuVector{Float32} + CuSparseMatrixCSR{Float32,Int32} (A30 ready)
  - solution_size(odeform) to size the initial state vector [was OS.function_size in bak/]

Run modes:
  julia tissue_armyheart.jl                    # CPU, all defaults
  SATO_USE_GPU=true julia tissue_armyheart.jl  # GPU 0
  SATO_USE_GPU=true CUDA_VISIBLE_DEVICES=2 julia tissue_armyheart.jl
  ARMYHEART_PATH=/path julia tissue_armyheart.jl
"""

# ============================================================================
# 0. Environment: activate ArmyHeart project (Thunderbolt, CUDA, LinearSolve, …)
# ============================================================================
const ARMYHEART_PATH = get(ENV, "ARMYHEART_PATH", "/tmp/ArmyHeart")
if isdir(joinpath(ARMYHEART_PATH, "Project.toml"))
    import Pkg
    Pkg.activate(ARMYHEART_PATH; io=devnull)
    Pkg.instantiate(; io=devnull)
    println("✓ ArmyHeart env: $ARMYHEART_PATH")
else
    @warn "ArmyHeart not found at $ARMYHEART_PATH — using default Julia env"
end

# ============================================================================
# 1. Imports (follow ArmyHeart's with_hetero_standalone.jl)
# ============================================================================
using Thunderbolt
import OrdinaryDiffEqOperatorSplitting as OS   # same as Thunderbolt exports OS
using LinearSolve
using LinearAlgebra
using Statistics
using Printf
using Adapt
using ProgressMeter

# ArmyHeart utility: steady_state_initializer! / distribute_initial_state!
const _AH_UTILS = joinpath(ARMYHEART_PATH, "src", "utils.jl")
if isfile(_AH_UTILS)
    include(_AH_UTILS)
    println("✓ ArmyHeart utils loaded")
else
    # Fallback inline implementation
    function steady_state_initializer!(u₀, f::GenericSplitFunction)
        odefun      = f.functions[2]
        ionic_model = odefun.ode
        dh          = f.functions[1].dh
        s₀flat      = @view u₀[f.solution_indices[2]]
        s₀          = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)))
        default_values = Thunderbolt.default_initial_state(ionic_model, nothing)
        for i in 1:Thunderbolt.num_states(ionic_model)
            s₀[:, i] .= default_values[i]
        end
    end
    function distribute_initial_state!(u₀, y, f::GenericSplitFunction)
        odefun      = f.functions[2]
        ionic_model = odefun.ode
        dh          = f.functions[1].dh
        s₀flat      = @view u₀[f.solution_indices[2]]
        s₀          = reshape(s₀flat, (ndofs(dh), Thunderbolt.num_states(ionic_model)))
        for i in 1:Thunderbolt.num_states(ionic_model)
            s₀[:, i] .= y[i]
        end
    end
    println("⚠ ArmyHeart utils fallback")
end

# Our Sato-Bers ArmyHeart ionic model
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================================
# 2. GPU / CPU setup (mirrors ArmyHeart 2D standalone)
# ============================================================================
const USE_GPU = parse(Bool, get(ENV, "SATO_USE_GPU", "false"))

if USE_GPU
    using CUDA
    @info "GPU mode: $(CUDA.name(CUDA.device()))"
end

# Float32 on GPU (A30 FP32 ≈ 10× FP64), Float64 on CPU
const _T = USE_GPU ? Float32 : Float64

# Types (following with_hetero_standalone.jl style)
const SolutionVectorType = USE_GPU ? CuVector{_T}                              : Vector{_T}
const SystemMatrixType   = USE_GPU ? CUDA.CUSPARSE.CuSparseMatrixCSR{_T,Int32} :
                                     ThreadedSparseMatrixCSR{_T,Int32}

# ============================================================================
# 3. Tissue parameters (Sato et al. 2025, Appendix Fig A2)
# ============================================================================
const L_MM      = 30.0          # tissue side length (mm) = 3 cm
const N_EL      = 60            # elements per side → 60×60 Q1 grid
const DX_MM     = L_MM / N_EL  # = 0.5 mm element size
const N_NX      = N_EL + 1      # = 61 nodes in x
const N_NY      = N_EL + 1      # = 61 nodes in y
const KAPPA     = 0.1           # isotropic conductivity D (mm²/ms)
const BCL       = 300.0         # basic cycle length (ms)
const DT₀       = 0.01          # timestep (ms) — "no touchy" per ArmyHeart
const DT_VIS    = 10.0          # ECG sample interval (ms)
const STIM_AMP  = 80.0          # stimulus amplitude (µA/µF)
const STIM_DUR  = 1.0           # stimulus duration (ms)
const CORNER_MM = DX_MM * 2     # pacing region: x,y < 1 mm

# ECG electrode position (outside tissue, opposite corner from pacing site)
const X_ELEC = L_MM + 5.0   # mm
const Y_ELEC = L_MM + 5.0   # mm

# ============================================================================
# 4. Build mesh + ODE form (semidiscrete monodomain system)
# ============================================================================

"""
    build_odeform(ionic) → GenericSplitFunction

60×60 Q1 Cartesian monodomain mesh.
Orthogonal-geometry speedup: constant-coefficient Laplacian → assembled once.
Follows ArmyHeart's with_hetero_standalone.jl structure.
"""
function build_odeform(ionic)
    T   = _T
    L   = T(L_MM)
    nel = N_EL

    # Quadrilateral (linear Q1) mesh — 60×60 elements, 61×61 nodes
    ep_mesh = generate_mesh(
        Quadrilateral, (nel, nel),
        Vec(T.((0.0, 0.0))),
        Vec(T.((L, L))),
    )
    ep_cs = CartesianCoordinateSystem(ep_mesh)

    # Isotropic conductivity κ (2×2 symmetric tensor)
    κ_tensor = ConstantCoefficient(
        SymmetricTensor{2, 2, T}((T(KAPPA), zero(T), T(KAPPA)))
    )

    # MonodomainModel: χ=1, Cₘ=1, κ, NoStim (stimulus lives in cell_rhs!)
    model = MonodomainModel(
        ConstantCoefficient(one(T)),
        ConstantCoefficient(one(T)),
        κ_tensor,
        NoStimulationProtocol(),
        ionic,
        :φₘ, :s,
    )

    # Semidiscretize with Q1 Lagrange elements
    odeform = semidiscretize(
        ReactionDiffusionSplit(model, ep_cs),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        ep_mesh,
    )
    return odeform
end

# ============================================================================
# 5. Solver (LieTrotterGodunov — ArmyHeart 2D standard)
# ============================================================================

"""
    build_solver() → OS.LieTrotterGodunov

BackwardEulerSolver (implicit diffusion) + AdaptiveForwardEulerSubstepper (ionic ODE)
wrapped in Lie-Trotter-Godunov operator splitting.
"""
function build_solver()
    T   = _T

    ep_stepper = BackwardEulerSolver(;
        inner_solver = KrylovJL_CG(;
            itmax   = 1000,
            atol    = T(1e-10),
            rtol    = T(1e-10),
            history = false,
        ),
        solution_vector_type = SolutionVectorType,
        system_matrix_type   = SystemMatrixType,
    )

    cell_stepper = AdaptiveForwardEulerSubstepper(;
        solution_vector_type = SolutionVectorType,
        reaction_threshold   = T(0.1),
        substeps             = 10,
    )

    return OS.LieTrotterGodunov((ep_stepper, cell_stepper))
end

# ============================================================================
# 6. Plonsey1964 pseudo-ECG setup
# ============================================================================

"""
    build_ecg_cache(odeform) → (Plonsey1964ECGGaussCache, n_nodes)

Assemble diffusion operator on CPU for the far-field ECG integral.
Uses Float64 for numerical accuracy regardless of GPU/CPU mode.
"""
function build_ecg_cache(odeform::GenericSplitFunction)
    dh   = odeform.functions[1].dh
    n    = ndofs(dh)

    κ_cpu = ConstantCoefficient(
        SymmetricTensor{2, 2, Float64}((KAPPA, 0.0, KAPPA))
    )
    strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    diff_op  = setup_assembled_operator(
        strategy,
        BilinearDiffusionIntegrator(κ_cpu, QuadratureRuleCollection(2), :φₘ),
        ThreadedSparseMatrixCSR{Float64, Int32},
        dh,
    )
    update_operator!(diff_op, 0.0)

    φ₀        = zeros(Float64, n)
    ecg_cache = Plonsey1964ECGGaussCache(diff_op, φ₀)
    return ecg_cache, n
end

# ============================================================================
# 7. Main simulation runner
# ============================================================================

"""
    run_tissue_sim(; tauf, av, N_CaL, ...) → (qt_intervals, ecg_t, ecg_v)

Run the 2D monodomain tissue simulation for one (τ_f, u) parameter point.
Returns beat-to-beat QT intervals (ms) and the ECG time/voltage trace.
"""
function run_tissue_sim(;
    tauf          :: Float64 = 52.0,
    av            :: Float64 = 3.0,
    N_CaL         :: Int     = 100_000,
    gna           :: Float64 = 12.0,
    gkr           :: Float64 = 0.0136,
    gks           :: Float64 = 0.0245,
    g_rel         :: Float64 = 75.0,
    prepace_beats :: Int     = 20,
    meas_beats    :: Int     = 60,
    verbose        :: Bool   = true,
)
    T = _T
    verbose && @printf("\n══ τ_f=%.1f  u=%.2f  N_CaL=%d  GPU=%s ══\n",
                       tauf, av, N_CaL, USE_GPU)

    # -- 7a. Ionic model (SatoBersArmyModel) --
    ionic_cpu = SatoBersArmyModel(;
        tauf      = tauf,  av        = av,
        N_CaL     = N_CaL, gna       = gna,
        gkr       = gkr,   gks       = gks,
        g_rel     = g_rel, nx        = N_NX,
        ny        = N_NY,  dx        = DX_MM,
        dy        = DX_MM, dt        = DT₀,
        bcl       = BCL,   stim_amp  = STIM_AMP,
        stim_dur  = STIM_DUR, corner_mm = CORNER_MM,
    )
    ionic = USE_GPU ? Adapt.adapt(CUDA.cu, ionic_cpu) : ionic_cpu

    # -- 7b. Semidiscrete monodomain ODE system --
    odeform = build_odeform(ionic)
    timestepper = build_solver()

    # -- 7c. Initial condition: resting state at all nodes --
    n_total = solution_size(odeform)              # ← correct API (not OS.function_size)
    u₀_cpu  = zeros(_T, n_total)
    steady_state_initializer!(u₀_cpu, odeform)
    u₀ = USE_GPU ? SolutionVectorType(u₀_cpu) : u₀_cpu

    # -- 7d. ECG cache (always CPU / Float64 for accuracy) --
    ecg_cache, n_nodes = build_ecg_cache(odeform)
    electrode = Vec{2, Float64}((X_ELEC, Y_ELEC))
    κₜ        = 1.0   # normalised torso conductivity

    # -- 7e. Pre-pacing: prepace_beats × BCL ms to reach limit cycle --
    verbose && @printf("  Pre-pacing %d beats…\n", prepace_beats)
    t0 = time()
    pre_tspan = (T(0), T(prepace_beats * BCL))
    pre_prob  = OS.OperatorSplittingProblem(odeform, copy(u₀), pre_tspan)
    pre_integ = OS.init(pre_prob, timestepper; dt=T(DT₀), verbose=false)

    pre_trange = T(0) : T(BCL) : T((prepace_beats - 1) * BCL)
    @showprogress for (_, _) in OS.TimeChoiceIterator(pre_integ, pre_trange) end
    u_ss = copy(pre_integ.u)
    verbose && @printf("  Pre-pacing done in %.1f s\n", time() - t0)

    # -- 7f. Measurement phase: meas_beats × BCL ms, sample ECG every DT_VIS ms --
    verbose && @printf("  Measuring %d beats (ECG sample every %.0f ms)…\n",
                       meas_beats, DT_VIS)
    t0 = time()

    meas_tspan = (T(0), T(meas_beats * BCL))
    meas_prob  = OS.OperatorSplittingProblem(odeform, u_ss, meas_tspan)
    meas_integ = OS.init(meas_prob, timestepper; dt=T(DT₀), verbose=false)

    heat_dofrange = odeform.solution_indices[1]   # φₘ DOF indices

    meas_trange = T(0) : T(DT_VIS) : T(meas_beats * BCL - DT_VIS)
    n_steps = length(meas_trange)
    ecg_t   = Float64.(collect(meas_trange))
    ecg_v   = zeros(Float64, n_steps)
    idx     = 0

    @showprogress for (u_, _) in OS.TimeChoiceIterator(meas_integ, meas_trange)
        idx += 1
        idx > n_steps && break

        # Copy φₘ to CPU as Float64 for ECG calculation
        φ_cpu = Float64.(Vector(u_[heat_dofrange]))
        update_ecg!(ecg_cache, φ_cpu)
        ecg_v[idx] = evaluate_ecg(ecg_cache, electrode, κₜ)
    end
    verbose && @printf("  Measurement done in %.1f s\n", time() - t0)

    # -- 7g. QT interval detection + QTVi --
    qt_intervals = detect_qt_intervals(ecg_v, meas_beats; dt=DT_VIS)
    if verbose
        n = length(qt_intervals)
        if n >= 5
            μ    = mean(qt_intervals)
            σ    = std(qt_intervals)
            qtvi = compute_qtvi(qt_intervals)
            @printf("  QT = %.2f ± %.3f ms  QTVi = %.4f  (n=%d)\n", μ, σ, qtvi, n)
        else
            @printf("  WARNING: only %d QT intervals detected\n", n)
        end
    end

    return qt_intervals, ecg_t, ecg_v
end

# ============================================================================
# 8. QT interval detection (tangent method on pseudo-ECG)
# ============================================================================

"""
    detect_qt_intervals(ecg, n_beats; dt) → Vector{Float64}

Detect QT intervals from pseudo-ECG trace using tangent method.
Each beat window is BCL/dt samples long.
"""
function detect_qt_intervals(
    ecg    :: Vector{Float64},
    n_beats :: Int;
    dt     :: Float64 = DT_VIS,
)
    QTs  = Float64[]
    step = round(Int, BCL / dt)
    for b in 0:(n_beats - 1)
        i0 = b * step + 1
        i1 = min((b + 1) * step, length(ecg))
        i1 - i0 < 20 && continue

        beat = ecg[i0:i1]
        n    = length(beat)
        w    = max(1, round(Int, 0.05n))

        # Baseline
        bl = 0.5 * (mean(beat[1:w]) + mean(beat[end-w+1:end]))

        # Q onset: first crossing of 20% of QRS peak above baseline
        qe   = round(Int, 0.40n)
        qe < 2 && continue
        qd   = abs.(beat[1:qe] .- bl)
        isempty(qd) && continue
        qpk  = argmax(qd)
        qs   = sign(beat[qpk] - bl)
        qt   = bl + 0.20 * qs * qd[qpk]
        qi   = 1
        for i in 1:qpk
            if (qs > 0 ? beat[i] > qt : beat[i] < qt)
                qi = i; break
            end
        end

        # T-wave end
        ts  = round(Int, 0.35n)
        te0 = round(Int, 0.95n)
        ts >= te0 && continue
        tseg = beat[ts:te0]
        isempty(tseg) && continue
        td   = abs.(tseg .- bl)
        tp   = argmax(td) + ts - 1
        tsgn = sign(beat[tp] - bl)
        tpk  = td[argmax(td)]
        tt   = bl + 0.05 * tsgn * tpk
        tei  = te0
        for i in te0:-1:tp
            i > length(beat) && continue
            if (tsgn > 0 ? beat[i] > tt : beat[i] < tt)
                tei = i; break
            end
        end

        qt_ms = (tei - qi) * dt
        50.0 < qt_ms < 450.0 && push!(QTs, qt_ms)
    end
    return QTs
end

"""
    compute_qtvi(qts) → Float64

QT variability index: log10(Var(QT)/Mean(QT)²) - log10(RR variance floor).
"""
function compute_qtvi(qts :: Vector{Float64})
    length(qts) < 5 && return NaN
    μ   = mean(qts)
    σ²  = max(var(qts), 1e-8)
    rr_var_floor = (1e-3 / BCL)^2  # ~constant pacing
    return log10(σ² / μ^2) - log10(rr_var_floor)
end

# ============================================================================
# 9. Main: τ_f sweep + u sweep → write CSVs (Fig A2C, A2D)
# ============================================================================

function main()
    println("="^72)
    println("2D Monodomain Tissue — ArmyHeart / Thunderbolt.jl")
    println("  Mesh:  $(N_EL)×$(N_EL) Q1 quads  →  $(N_NX)×$(N_NY) nodes")
    println("  GPU:   $USE_GPU   (precision: $_T)")
    println("  κ:     $KAPPA mm²/ms   dt₀=$DT₀ ms   BCL=$BCL ms")
    println("  ECG electrode: ($X_ELEC, $Y_ELEC) mm")
    println("="^72)

    out = joinpath(@__DIR__, "..", "figures", "data")
    mkpath(out)

    prepace = 20
    meas    = 60
    ecg_saved = Dict{String, Tuple{Vector{Float64},Vector{Float64}}}()

    # ── τ_f sweep (Fig A2C) ──────────────────────────────────────────────────
    println("\n=== τ_f sweep → QTVi (Fig A2C) ===")
    tauf_vals = [20.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 52.0]
    open(joinpath(out, "tissue2d_tauf_qtv.csv"), "w") do io
        println(io, "tauf,qt_mean,qt_std,qtvi,n_beats")
        for tf in tauf_vals
            qts, et, ev = run_tissue_sim(;
                tauf=tf, av=3.0, N_CaL=100_000,
                prepace_beats=prepace, meas_beats=meas,
            )
            n = length(qts)
            if n < 5
                println(io, "$tf,NaN,NaN,NaN,$n")
            else
                μ    = mean(qts); σ = std(qts); qtvi = compute_qtvi(qts)
                println(io, "$tf,$μ,$σ,$qtvi,$n")
            end
            flush(io)
            tf in [30.0, 45.0, 52.0] && (ecg_saved["tauf$(round(Int,tf))"] = (et, ev))
        end
    end
    println("  → tissue2d_tauf_qtv.csv written")

    # ── u sweep (Fig A2D) ────────────────────────────────────────────────────
    println("\n=== u sweep → QTVi (Fig A2D) ===")
    u_vals = [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5]
    open(joinpath(out, "tissue2d_u_qtv.csv"), "w") do io
        println(io, "u,qt_mean,qt_std,qtvi,n_beats")
        for uv in u_vals
            qts, et, ev = run_tissue_sim(;
                tauf=35.0, av=uv, N_CaL=100_000,
                prepace_beats=prepace, meas_beats=meas,
            )
            n = length(qts)
            if n < 5
                println(io, "$uv,NaN,NaN,NaN,$n")
            else
                μ    = mean(qts); σ = std(qts); qtvi = compute_qtvi(qts)
                println(io, "$uv,$μ,$σ,$qtvi,$n")
            end
            flush(io)
            uv in [5.0, 8.0, 9.5] && (ecg_saved["u$(uv)"] = (et, ev))
        end
    end
    println("  → tissue2d_u_qtv.csv written")

    # ── Save ECG traces ───────────────────────────────────────────────────────
    if !isempty(ecg_saved)
        ks = sort(collect(keys(ecg_saved)))
        open(joinpath(out, "tissue2d_ecg_traces.csv"), "w") do io
            println(io, "t_ms," * join(ks, ","))
            n = minimum(length(v[1]) for v in values(ecg_saved))
            for i in 1:n
                t_str = string(round(ecg_saved[ks[1]][1][i], digits=2))
                vals  = join([string(round(ecg_saved[k][2][i], digits=8)) for k in ks], ",")
                println(io, "$t_str,$vals")
            end
        end
        println("  → tissue2d_ecg_traces.csv written")
    end

    println("\n✓ 2D tissue simulation complete — see figures/data/tissue2d_*.csv")
    return nothing
end

# ── Entry point ──
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
