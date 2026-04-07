#!/usr/bin/env julia
"""
tissue_armyheart.jl — 2D monodomain tissue simulation using ArmyHeart/Thunderbolt.jl

Reproduces Appendix Fig A2 from Sato et al. (2025):
  - 3 cm × 3 cm, 60×60 Q1 Quadrilateral elements (61×61 = 3721 nodes)
  - Corner pacing (x < 1 mm, y < 1 mm), baked into cell_rhs! of SatoBersArmyModel
  - Pseudo-ECG via Thunderbolt Plonsey1964ECGGaussCache
  - QTVi vs τ_f (Fig A2C) and QTVi vs u (Fig A2D)
  - Stochastic ICaL gating (N_CaL=100,000) → beat-to-beat APD variability

Run modes:
  julia tissue_armyheart.jl              # CPU Float64
  SATO_USE_GPU=true julia tissue_armyheart.jl    # GPU Float32 on A30
  ARMYHEART_PATH=/path julia tissue_armyheart.jl
"""

# ============================================================================
# 0. Environment: activate ArmyHeart project
# ============================================================================
const ARMYHEART_PATH = get(ENV, "ARMYHEART_PATH", "/tmp/armyheart")
if isdir(joinpath(ARMYHEART_PATH, "Project.toml"))
    import Pkg
    Pkg.activate(ARMYHEART_PATH; io=devnull)
    Pkg.instantiate(; io=devnull)
    println("✓ ArmyHeart env: $ARMYHEART_PATH")
end

# ============================================================================
# 1. Imports
# ============================================================================
using Thunderbolt
using LinearSolve: KrylovJL_CG
using LinearAlgebra
using Statistics
using Printf
using Adapt

const OS = Thunderbolt.OS

# ============================================================================
# 2. GPU / CPU setup
# ============================================================================
const USE_GPU = parse(Bool, get(ENV, "SATO_USE_GPU", "false"))

if USE_GPU
    using CUDA
    @info "GPU mode: $(CUDA.name(CUDA.device()))"
end

const _T = USE_GPU ? Float32 : Float64

SolutionVectorType = USE_GPU ? CuVector{_T}                              : Vector{_T}
SystemMatrixType   = USE_GPU ? CUDA.CUSPARSE.CuSparseMatrixCSR{_T,Int32} :
                               Thunderbolt.ThreadedSparseMatrixCSR{_T,Int32}

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
const DT₀       = 0.01          # timestep (ms)
const DT_VIS    = 5.0           # ECG sample interval (ms) — fine enough for QT
const STIM_AMP  = 80.0          # stimulus amplitude (µA/µF)
const STIM_DUR  = 1.0           # stimulus duration (ms)
const CORNER_MM = DX_MM * 2     # pacing region: x,y < 1 mm

# ECG electrode at opposite corner from pacing
const X_ELEC = L_MM + 5.0
const Y_ELEC = L_MM + 5.0

# Our SatoBers model
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

# ============================================================================
# 4. Build mesh + ODE form
# ============================================================================
function build_odeform(ionic)
    T   = _T
    L   = T(L_MM)
    nel = N_EL

    ep_mesh = generate_mesh(
        Quadrilateral, (nel, nel),
        Vec(T.((0.0, 0.0))),
        Vec(T.((L, L))),
    )
    ep_cs = CartesianCoordinateSystem(ep_mesh)

    κ_tensor = ConstantCoefficient(
        SymmetricTensor{2, 2, T}((T(KAPPA), zero(T), T(KAPPA)))
    )

    model = MonodomainModel(
        ConstantCoefficient(one(T)),
        ConstantCoefficient(one(T)),
        κ_tensor,
        NoStimulationProtocol(),
        ionic,
        :φₘ, :s,
    )

    odeform = semidiscretize(
        ReactionDiffusionSplit(model, ep_cs),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
        ep_mesh,
    )
    return odeform
end

# ============================================================================
# 5. Solver
# ============================================================================
function build_solver()
    T = _T

    ep_stepper = BackwardEulerSolver(;
        inner_solver = KrylovJL_CG(;
            itmax   = 2000,
            atol    = T(1e-8),
            rtol    = T(1e-8),
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
# 6. Plonsey1964 pseudo-ECG setup (always CPU / Float64 for accuracy)
# ============================================================================
function build_ecg_cache(odeform)
    dh = odeform.functions[1].dh
    n  = ndofs(dh)

    κ_cpu = ConstantCoefficient(
        SymmetricTensor{2, 2, Float64}((KAPPA, 0.0, KAPPA))
    )
    strategy = Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice())
    diff_op = Thunderbolt.setup_assembled_operator(
        strategy,
        Thunderbolt.BilinearDiffusionIntegrator(κ_cpu, QuadratureRuleCollection(2), :φₘ),
        Thunderbolt.ThreadedSparseMatrixCSR{Float64, Int32},
        dh,
    )
    Thunderbolt.update_operator!(diff_op, 0.0)

    φ₀        = zeros(Float64, n)
    ecg_cache = Thunderbolt.Plonsey1964ECGGaussCache(diff_op, φ₀)
    return ecg_cache, n
end

# ============================================================================
# 7. Initial condition setup
# ============================================================================
function initialize_u0!(u₀::AbstractVector, odeform, ionic_cpu)
    dh       = odeform.functions[1].dh
    n_nodes  = ndofs(dh)
    ionic_dofs = odeform.solution_indices[2]
    u0_cell  = Thunderbolt.default_initial_state(ionic_cpu, nothing)

    u₀_cpu = USE_GPU ? Array(u₀) : u₀
    s0 = reshape(view(u₀_cpu, ionic_dofs), (n_nodes, Thunderbolt.num_states(ionic_cpu)))
    for i in 1:Thunderbolt.num_states(ionic_cpu)
        s0[:, i] .= _T(u0_cell[i])
    end

    if USE_GPU
        copyto!(u₀, u₀_cpu)
    end
    return u₀
end

# ============================================================================
# 8. Main simulation runner
# ============================================================================
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

    # -- 8a. Ionic model (CPU first, then GPU adapt if needed) --
    ionic_cpu = SatoBersArmyModel(;
        tauf      = tauf,  av        = av,
        N_CaL     = N_CaL, gna       = gna,
        gkr       = gkr,   gks       = gks,
        g_rel     = g_rel, nx        = N_NX,
        ny        = N_NY,  dx        = DX_MM,
        dy        = DX_MM, dt        = Float64(DT₀),
        bcl       = BCL,   stim_amp  = STIM_AMP,
        stim_dur  = STIM_DUR, corner_mm = CORNER_MM,
    )
    ionic = USE_GPU ? Adapt.adapt(CuArray, ionic_cpu) : ionic_cpu

    # -- 8b. Semidiscrete ODE system --
    odeform     = build_odeform(ionic)
    timestepper = build_solver()

    # -- 8c. Initial condition --
    n_total = Thunderbolt.solution_size(odeform)
    u₀ = USE_GPU ? SolutionVectorType(zeros(_T, n_total)) : zeros(_T, n_total)
    initialize_u0!(u₀, odeform, ionic_cpu)

    # -- 8d. ECG cache --
    ecg_cache, n_nodes = build_ecg_cache(odeform)
    electrode = Vec{2, Float64}((X_ELEC, Y_ELEC))
    κₜ        = 1.0

    heat_dofrange = odeform.solution_indices[1]

    # -- 8e. Pre-pacing to limit cycle --
    verbose && @printf("  Pre-pacing %d beats…\n", prepace_beats)
    t0 = time()
    pre_tspan = (T(0), T(prepace_beats * BCL))
    pre_prob  = OS.OperatorSplittingProblem(odeform, copy(u₀), pre_tspan)
    pre_integ = Thunderbolt.init(pre_prob, timestepper; dt=T(DT₀))

    pre_trange = T(0) : T(BCL) : T((prepace_beats - 1) * BCL)
    for (_, _) in TimeChoiceIterator(pre_integ, pre_trange) end
    u_ss = copy(pre_integ.u)
    verbose && @printf("  Pre-pacing done in %.1f s\n", time() - t0)

    # -- 8f. Measurement phase: sample ECG every DT_VIS ms --
    verbose && @printf("  Measuring %d beats (ECG every %.0f ms)…\n",
                       meas_beats, DT_VIS)
    t0 = time()

    meas_tspan = (T(0), T(meas_beats * BCL))
    meas_prob  = OS.OperatorSplittingProblem(odeform, u_ss, meas_tspan)
    meas_integ = Thunderbolt.init(meas_prob, timestepper; dt=T(DT₀))

    meas_trange = T(0) : T(DT_VIS) : T(meas_beats * BCL - DT_VIS)
    n_steps = length(meas_trange)
    ecg_t   = Float64.(collect(meas_trange))
    ecg_v   = zeros(Float64, n_steps)
    idx     = 0

    for (u_, _) in TimeChoiceIterator(meas_integ, meas_trange)
        idx += 1
        idx > n_steps && break
        φ_cpu = Float64.(Vector(u_[heat_dofrange]))
        Thunderbolt.update_ecg!(ecg_cache, φ_cpu)
        ecg_v[idx] = Thunderbolt.evaluate_ecg(ecg_cache, electrode, κₜ)
    end
    verbose && @printf("  Measurement done in %.1f s\n", time() - t0)

    # -- 8g. QT interval detection + QTVi --
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
# 9. QT interval detection
# ============================================================================
function detect_qt_intervals(
    ecg     :: Vector{Float64},
    n_beats :: Int;
    dt      :: Float64 = DT_VIS,
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

        # Q onset: first time signal exceeds 20% of QRS peak
        qe  = round(Int, 0.40n)
        qe < 2 && continue
        qd  = abs.(beat[1:qe] .- bl)
        isempty(qd) && continue
        qpk = argmax(qd)
        qs  = sign(beat[qpk] - bl)
        qt  = bl + 0.20 * qs * qd[qpk]
        qi  = 1
        for i in 1:qpk
            if (qs > 0 ? beat[i] > qt : beat[i] < qt)
                qi = i; break
            end
        end

        # T-wave end (tangent method approximation)
        ts   = round(Int, 0.35n)
        te0  = round(Int, 0.95n)
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

function compute_qtvi(qts :: Vector{Float64})
    length(qts) < 5 && return NaN
    μ   = mean(qts)
    σ²  = max(var(qts), 1e-8)
    rr_var_floor = (1e-3 / BCL)^2
    return log10(σ² / μ^2) - log10(rr_var_floor)
end

# ============================================================================
# 10. Main: τ_f and u parameter sweeps
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
            if tf in [30.0, 45.0, 52.0]
                ecg_saved["tauf$(round(Int,tf))"] = (et, ev)
            end
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
            if uv in [5.0, 8.0, 9.5]
                ecg_saved["u$(uv)"] = (et, ev)
            end
        end
    end
    println("  → tissue2d_u_qtv.csv written")

    # ── Save ECG traces ────────────────────────────────────────────────────
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

    println("\n✓ 2D tissue simulation complete")
    return nothing
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
