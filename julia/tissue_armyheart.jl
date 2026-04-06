#!/usr/bin/env julia
"""
tissue_armyheart.jl — 2D monodomain tissue simulation using ArmyHeart

Reproduces Appendix Fig A2 from Sato et al. (2025):
  - 3 cm × 3 cm, 60×60 Q1 Quadrilateral elements (61×61 = 3721 nodes)
  - Corner pacing at x < 1 mm, y < 1 mm (baked into cell_rhs! of SatoBersArmyModel)
  - Pseudo-ECG via Thunderbolt's Plonsey1964ECGGaussCache (Plonsey 1964 far-field)
  - QTVi vs τ_f (Fig A2C) and QTVi vs u (Fig A2D)
  - Stochastic ICaL gating (N_CaL=100,000) → beat-to-beat APD variability

# Architecture (follows ArmyHeart 2D experiment pattern)
  - Environment: ArmyHeart Project.toml (provides Thunderbolt, CUDA, LinearSolve, etc.)
  - Mesh: generate_mesh(Quadrilateral, (60,60), ...) — regular Cartesian
    "Orthogonal geometry" speedup: constant-coeff Laplacian → stiffness assembled once
  - Model: MonodomainModel + NoStimulationProtocol (stimulus in cell_rhs!)
  - Splitter: OS.LieTrotterGodunov(BackwardEulerSolver, AdaptiveForwardEulerSubstepper)
  - GPU: CuVector{Float32} + CuSparseMatrixCSR{Float32,Int32} (A30 ready)

# How to run
  julia tissue_armyheart.jl                                      # CPU
  SATO_USE_GPU=true julia tissue_armyheart.jl                    # GPU 0
  SATO_USE_GPU=true CUDA_VISIBLE_DEVICES=2 julia ...             # GPU 2
  ARMYHEART_PATH=/path/to/ArmyHeart julia tissue_armyheart.jl    # custom AH path

# Batch (multi-GPU τ_f / u sweep)
  julia tissue_armyheart_batch.jl                                # see that script
"""

# ============================================================================
# Environment: activate ArmyHeart (provides Thunderbolt, CUDA, LinearSolve…)
# ============================================================================
const ARMYHEART_PATH = get(ENV, "ARMYHEART_PATH", "/tmp/ArmyHeart")

if isdir(joinpath(ARMYHEART_PATH, "Project.toml"))
    import Pkg
    Pkg.activate(ARMYHEART_PATH; io=devnull)
    Pkg.instantiate(; io=devnull)
    println("✓ ArmyHeart environment: $ARMYHEART_PATH")
else
    @warn "ArmyHeart not found at $ARMYHEART_PATH — using default Julia env. " *
          "Set ARMYHEART_PATH=/path/to/ArmyHeart"
end

# ============================================================================
# Imports — follow ArmyHeart conventions
# ============================================================================
using Thunderbolt
import Thunderbolt.OS               # OrdinaryDiffEqOperatorSplitting
import Thunderbolt: OS as TbOS      # alias for clarity
using LinearSolve                   # KrylovJL_CG
using Statistics
using Printf
using Adapt

# ArmyHeart utility functions (distribute_initial_state!, evaluate_at_grid_nodes)
const AH_UTILS = joinpath(ARMYHEART_PATH, "src", "utils.jl")
if isfile(AH_UTILS)
    include(AH_UTILS)
    println("✓ ArmyHeart utils loaded")
else
    # Inline the functions we need if ArmyHeart not present
    function distribute_initial_state!(u₀, y, f)
        odefun    = f.functions[2]
        ionic_model = odefun.ode
        ionic_dofrange = f.solution_indices[2]
        dh = f.functions[1].dh
        s₀flat = @view u₀[ionic_dofrange]
        s₀ = reshape(s₀flat, (Thunderbolt.Ferrite.ndofs(dh), Thunderbolt.num_states(ionic_model)))
        for i in 1:Thunderbolt.num_states(ionic_model)
            s₀[:, i] .= y[i]
        end
    end
    function steady_state_initializer!(u₀, f)
        odefun    = f.functions[2]
        ionic_model = odefun.ode
        default_values = Thunderbolt.default_initial_state(ionic_model, nothing)
        distribute_initial_state!(u₀, default_values, f)
    end
    println("⚠ ArmyHeart utils not found — using inline fallbacks")
end

# Our Sato-Bers ArmyHeart model (follows LuoRudy.jl pattern)
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================================
# GPU / CPU toggle
# ============================================================================
const USE_GPU = parse(Bool, get(ENV, "SATO_USE_GPU", "false"))

if USE_GPU
    using CUDA
    @info "GPU mode: $(CUDA.name(CUDA.device()))"
end

# Use Float32 on GPU (A30 FP32 is 10× faster than FP64), Float64 on CPU
const _T = USE_GPU ? Float32 : Float64

# Vector / matrix types
_SolutionVectorType() = USE_GPU ? CuVector{_T}                          : Vector{_T}
_SystemMatrixType()   = USE_GPU ? CUDA.CUSPARSE.CuSparseMatrixCSR{_T, Int32} :
                                  Thunderbolt.ThreadedSparseMatrixCSR{_T, Int32}

# ============================================================================
# Tissue parameters (Sato et al. 2025, Appendix Fig A2)
# ============================================================================
const L_MM   = 30.0          # Tissue side length (mm) = 3 cm
const N_EL   = 60            # Elements per side  → 60×60 Q1 elements
const DX_MM  = L_MM / N_EL  # = 0.5 mm node spacing
const N_NX   = N_EL + 1      # = 61 nodes in x
const N_NY   = N_EL + 1      # = 61 nodes in y

# Isotropic conductivity κ = D = 0.1 mm²/ms
# χ = Cₘ = 1 (dimensionless normalisation used by Sato 2025)
const KAPPA  = _T(0.1)

# Far-field ECG electrode: corner diagonally opposite the pacing site
const X_ELEC = _T(L_MM + 2.5)
const Y_ELEC = _T(L_MM + 2.5)

# Simulation timing
const BCL       = 300.0   # ms  basic cycle length
const DT₀       = 0.01    # ms  time step (ArmyHeart default — "no touchy")
const STIM_AMP  = 80.0    # µA/µF
const STIM_DUR  = 1.0     # ms
const CORNER_MM = DX_MM * 2  # pacing region: x,y < 1 mm

# ============================================================================
# Build mesh + model (shared across all parameter points)
# ============================================================================

"""
    build_odeform(ionic) → odeform

Assemble the semidiscrete monodomain ODE system for one parameter point.
Mesh is 60×60 Q1 Cartesian grid — constant-coefficient Laplacian.
"""
function build_odeform(ionic)
    T = _T

    # -- 60×60 Q1 (Quadrilateral) mesh — "orthogonal geometry" Cartesian grid
    ep_mesh = generate_mesh(
        Quadrilateral,
        (N_EL, N_EL),
        Thunderbolt.Vec(T.((0.0, 0.0))),
        Thunderbolt.Vec(T.((L_MM, L_MM))),
    )
    ep_cs = CartesianCoordinateSystem(ep_mesh)

    # Isotropic conductivity tensor κI (2×2 symmetric)
    κ_tensor = ConstantCoefficient(
        SymmetricTensor{2, 2, T}((KAPPA, zero(T), KAPPA))
    )

    # MonodomainModel: χ=1, Cₘ=1, κ, NoStim (stimulus inside cell_rhs!)
    model = MonodomainModel(
        ConstantCoefficient(one(T)),
        ConstantCoefficient(one(T)),
        κ_tensor,
        NoStimulationProtocol(),
        ionic,
        :φₘ,
        :s,
    )

    # Semidiscretize with Q1 Lagrange elements
    odeform = semidiscretize(
        ReactionDiffusionSplit(model, ep_cs),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        ep_mesh,
    )

    return odeform
end

"""
    build_solver() → timestepper

BackwardEulerSolver (implicit diffusion) + AdaptiveForwardEulerSubstepper (cell ODEs)
wrapped in LieTrotterGodunov splitting — exact ArmyHeart 2D pattern.
"""
function build_solver()
    T = _T
    SolutionVectorType = _SolutionVectorType()
    SystemMatrixType   = _SystemMatrixType()

    ep_stepper = BackwardEulerSolver(;
        inner_solver = LinearSolve.KrylovJL_CG(;
            itmax   = 1000,
            atol    = T(1e-10),
            rtol    = T(1e-10),
            history = false,
        ),
        solution_vector_type = typeof(SolutionVectorType),
        system_matrix_type   = typeof(SystemMatrixType),
    )

    cell_stepper = AdaptiveForwardEulerSubstepper(;
        solution_vector_type = typeof(SolutionVectorType),
        reaction_threshold   = T(0.1),
        substeps             = 10,
    )

    return OS.LieTrotterGodunov((ep_stepper, cell_stepper))
end

# ============================================================================
# ECG setup (Plonsey 1964 far-field, via Thunderbolt)
# ============================================================================

"""
    build_ecg_cache(odeform) → (cache, dh_for_ecg)

Set up Plonsey1964ECGGaussCache for the tissue mesh.
The diffusion operator is assembled on CPU (even in GPU runs).
"""
function build_ecg_cache(odeform)
    T = Float64   # ECG always double-precision on CPU

    dh = odeform.functions[1].dh
    n_nodes = Thunderbolt.Ferrite.ndofs(dh)

    κ_coeff_cpu = ConstantCoefficient(SymmetricTensor{2, 2, T}((_T(KAPPA), zero(T), _T(KAPPA))))

    # CPU-side assembled diffusion operator (stiffness matrix for Plonsey formula)
    cpu_strategy = SequentialAssemblyStrategy(SequentialCPUDevice())
    diff_op = setup_assembled_operator(
        cpu_strategy,
        BilinearDiffusionIntegrator(κ_coeff_cpu, QuadratureRuleCollection(2), :φₘ),
        Thunderbolt.ThreadedSparseMatrixCSR{T, Int32},
        dh,
    )
    update_operator!(diff_op, 0.0)

    φ₀ = zeros(T, n_nodes)
    ecg_cache = Plonsey1964ECGGaussCache(diff_op, φ₀)
    return ecg_cache, n_nodes
end

# ============================================================================
# Main simulation runner
# ============================================================================

"""
    run_tissue_sim(; tauf, av, N_CaL, ...) → (qt_intervals, ecg_trace)

Run the 2D monodomain tissue simulation for one (τ_f, u) parameter point.
Returns beat-to-beat QT intervals (ms) and the full ECG trace.
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

    # -- Ionic model (Sato-Bers, follows ArmyHeart LuoRudy.jl pattern) --
    ionic_cpu = SatoBersArmyModel(;
        tauf      = tauf,
        av        = av,
        N_CaL     = N_CaL,
        gna       = gna,
        gkr       = gkr,
        gks       = gks,
        g_rel     = g_rel,
        nx        = N_NX,
        ny        = N_NY,
        dx        = DX_MM,
        dy        = DX_MM,
        dt        = DT₀,
        bcl       = BCL,
        stim_amp  = STIM_AMP,
        stim_dur  = STIM_DUR,
        corner_mm = CORNER_MM,
    )
    ionic = USE_GPU ? Adapt.adapt(CUDA.cu, ionic_cpu) : ionic_cpu

    # -- Build semidiscrete form --
    odeform = build_odeform(ionic)

    # -- Initial state: fill all nodes with default resting state --
    n_total = OS.function_size(odeform)
    u₀_cpu = zeros(T, n_total)
    steady_state_initializer!(u₀_cpu, odeform)
    u₀ = USE_GPU ? _SolutionVectorType()(u₀_cpu) : u₀_cpu

    # -- ECG cache (CPU side, double precision) --
    ecg_cache, n_nodes = build_ecg_cache(odeform)
    electrode = Thunderbolt.Vec(Float64.((X_ELEC, Y_ELEC)))
    κₜ        = 1.0   # ratio σᵢ / (4π σₑ) normalised to 1

    # -- Solver: LieTrotterGodunov(BackwardEuler, AdaptiveFE) --
    timestepper = build_solver()

    # ── Pre-pacing: run prepace_beats × BCL ms to reach limit cycle ──
    verbose && @printf("  Pre-pacing %d beats (%.0f ms)…\n",
                       prepace_beats, prepace_beats * BCL)
    t0_pre = time()
    pre_prob  = OS.OperatorSplittingProblem(
        odeform, copy(u₀), (T(0), T(prepace_beats * BCL))
    )
    pre_integ = OS.init(pre_prob, timestepper; dt=T(DT₀), verbose=false)
    # Advance quietly (no output collection needed)
    @showprogress for (_, _) in OS.TimeChoiceIterator(
            pre_integ, T(0) : T(BCL) : T((prepace_beats-1)*BCL))
    end
    u_ss = copy(pre_integ.u)
    verbose && @printf("  Pre-pacing done in %.1f s\n", time() - t0_pre)

    # ── Measurement phase: meas_beats × BCL ms, collect ECG ──
    verbose && @printf("  Measuring %d beats (%.0f ms)…\n",
                       meas_beats, meas_beats * BCL)
    t0_meas = time()
    meas_prob  = OS.OperatorSplittingProblem(
        odeform, u_ss, (T(0), T(meas_beats * BCL))
    )
    meas_integ = OS.init(meas_prob, timestepper; dt=T(DT₀), verbose=false)

    heat_dofrange = odeform.solution_indices[1]   # φₘ DOF indices

    n_steps   = round(Int, meas_beats * BCL / DT₀)
    ecg_trace = zeros(Float64, n_steps)
    step_idx  = 0

    for (u_, _) in OS.TimeChoiceIterator(
            meas_integ,
            T(0) : T(DT₀) : T(meas_beats * BCL - DT₀))
        step_idx += 1
        step_idx > n_steps && break

        # Extract φₘ on CPU (float64 for ECG accuracy)
        φ_cpu = Float64.(Vector(u_[heat_dofrange]))
        update_ecg!(ecg_cache, φ_cpu)
        ecg_trace[step_idx] = evaluate_ecg(ecg_cache, electrode, κₜ)
    end
    verbose && @printf("  Measurement done in %.1f s\n", time() - t0_meas)

    # ── QT detection + QTVi ──
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
# QT interval detection (Tangent method on the pseudo-ECG)
# ============================================================================

function detect_qt_intervals(
    ecg :: Vector{Float64},
    n_beats :: Int;
    dt  :: Float64 = DT₀,
)
    QTs  = Float64[]
    step = round(Int, BCL / dt)
    for b in 0:(n_beats - 1)
        i0 = b*step + 1
        i1 = min((b+1)*step, length(ecg))
        i1 - i0 < 50 && continue

        beat = ecg[i0:i1]
        n    = length(beat)
        w    = max(1, round(Int, 0.03n))
        bl   = 0.5 * (mean(beat[1:w]) + mean(beat[end-w+1:end]))

        # Q onset
        qe   = round(Int, 0.40n)
        qd   = abs.(beat[1:qe] .- bl)
        isempty(qd) && continue
        qp   = argmax(qd)
        qs   = sign(beat[qp] - bl)
        qt   = bl + 0.20 * qs * qd[qp]
        qi   = 1
        for i in 1:qp
            if (qs > 0 ? beat[i] > qt : beat[i] < qt)
                qi = i; break
            end
        end

        # T-wave end
        ts  = round(Int, 0.35n)
        te0 = round(Int, 0.95n)
        ts >= te0 && continue
        td  = abs.(beat[ts:te0] .- bl)
        tp  = argmax(td) + ts - 1
        tsgn = sign(beat[tp] - bl)
        tpk  = td[argmax(td)]
        tt   = bl + 0.05 * tsgn * tpk
        tei  = te0
        for i in te0:-1:tp
            if (tsgn > 0 ? beat[i] > tt : beat[i] < tt)
                tei = i; break
            end
        end

        qt_ms = (tei - qi) * dt
        50.0 < qt_ms < 400.0 && push!(QTs, qt_ms)
    end
    return QTs
end

function compute_qtvi(qts :: Vector{Float64})
    length(qts) < 5 && return NaN
    μ  = mean(qts)
    σ² = max(var(qts), 1e-8)
    # Floor RR variance to avoid -Inf when BCL is constant
    rr_var_floor = 1e-6 / BCL^2
    return log10(σ² / μ^2) - log10(rr_var_floor)
end

# ============================================================================
# Main: τ_f sweep + u sweep → write CSVs (Fig A2C, A2D)
# ============================================================================

function main()
    println("="^72)
    println("2D Monodomain Tissue — ArmyHeart / Thunderbolt.jl")
    println("  Mesh:     $(N_EL)×$(N_EL) Q1 quads  ($(N_NX)×$(N_NY) nodes)")
    println("  GPU:      $USE_GPU   (precision: $_T)")
    println("  κ:        $(KAPPA) mm²/ms   dt₀=$(DT₀) ms   BCL=$(BCL) ms")
    println("  Electrode: ($(X_ELEC), $(Y_ELEC)) mm")
    println("="^72)

    out = joinpath(@__DIR__, "..", "figures", "data")
    mkpath(out)

    prepace = 20
    meas    = 60
    ecg_saved = Dict{String, Vector{Float64}}()

    # ── τ_f sweep (Fig A2C) ──────────────────────────────────────────────────
    println("\n=== τ_f sweep → QTVi (Fig A2C) ===")
    tauf_vals = [20.0, 30.0, 35.0, 40.0, 45.0, 48.0, 50.0, 52.0]
    open(joinpath(out, "tissue2d_tauf_qtv.csv"), "w") do io
        println(io, "tauf,qt_mean,qt_std,qtvi,n_beats")
        for tf in tauf_vals
            qts, ecg = run_tissue_sim(;
                tauf=tf, av=3.0, N_CaL=100_000,
                prepace_beats=prepace, meas_beats=meas,
            )
            if length(qts) < 5
                println(io, "$tf,NaN,NaN,NaN,$(length(qts))")
            else
                μ = mean(qts); σ = std(qts); qtvi = compute_qtvi(qts)
                println(io, "$tf,$μ,$σ,$qtvi,$(length(qts))")
            end
            flush(io)
            tf in [30.0, 45.0, 52.0] && (ecg_saved["tauf$(round(Int,tf))"] = ecg)
        end
    end
    println("  → tissue2d_tauf_qtv.csv written")

    # ── u sweep (Fig A2D) ────────────────────────────────────────────────────
    println("\n=== u sweep → QTVi (Fig A2D) ===")
    u_vals = [3.0, 5.0, 7.0, 8.0, 8.5, 9.0, 9.5]
    open(joinpath(out, "tissue2d_u_qtv.csv"), "w") do io
        println(io, "u,qt_mean,qt_std,qtvi,n_beats")
        for uv in u_vals
            qts, ecg = run_tissue_sim(;
                tauf=35.0, av=uv, N_CaL=100_000,
                prepace_beats=prepace, meas_beats=meas,
            )
            if length(qts) < 5
                println(io, "$uv,NaN,NaN,NaN,$(length(qts))")
            else
                μ = mean(qts); σ = std(qts); qtvi = compute_qtvi(qts)
                println(io, "$uv,$μ,$σ,$qtvi,$(length(qts))")
            end
            flush(io)
            uv in [5.0, 8.0, 9.5] && (ecg_saved["u$(uv)"] = ecg)
        end
    end
    println("  → tissue2d_u_qtv.csv written")

    # ── Save ECG traces ───────────────────────────────────────────────────────
    if !isempty(ecg_saved)
        ks = sort(collect(keys(ecg_saved)))
        open(joinpath(out, "tissue2d_ecg_traces.csv"), "w") do io
            println(io, "t_ms," * join(ks, ","))
            n = minimum(length(v) for v in values(ecg_saved))
            for i in 1:n
                t_str = string(round((i-1)*DT₀, digits=3))
                vals  = join([string(ecg_saved[k][i]) for k in ks], ",")
                println(io, "$t_str,$vals")
            end
        end
        println("  → tissue2d_ecg_traces.csv written")
    end

    println("\n✓ 2D tissue simulation complete.")
    return nothing
end

# Only run main() when executed directly (not when included from batch script)
if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
