#!/usr/bin/env julia
#"""
tissue_quick.jl — Fast 2D tissue simulation (20×20 mesh) for Fig A2 demo.

Demonstrates QTVi vs τ_f trend on a smaller mesh for faster results.
#"""
using Thunderbolt
using LinearSolve: KrylovJL_CG
using LinearAlgebra; using Statistics; using Printf; using Adapt

const OS = Thunderbolt.OS
const L_MM = 10.0; const N_EL = 20
const DX_MM = L_MM / N_EL; const N_NX = N_EL + 1; const N_NY = N_EL + 1
const KAPPA = 0.1; const BCL = 300.0; const DT₀ = 0.01
const DT_VIS = 5.0; const STIM_AMP = 80.0; const STIM_DUR = 1.0
const CORNER_MM = DX_MM * 2
const X_ELEC = L_MM + 5.0; const Y_ELEC = L_MM + 5.0

include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

function run_sim(tauf, av; N_CaL=100000, prepace=5, meas=20)
    @printf("  τ_f=%.1f u=%.1f\n", tauf, av)
    ionic = SatoBersArmyModel(;
        tauf=tauf, av=av, N_CaL=N_CaL,
        nx=N_NX, ny=N_NY, dx=DX_MM, dy=DX_MM,
        dt=DT₀, bcl=BCL, stim_amp=STIM_AMP, stim_dur=STIM_DUR, corner_mm=CORNER_MM,
    )
    ep_mesh = generate_mesh(Quadrilateral, (N_EL, N_EL), Vec((0.0,0.0)), Vec((L_MM, L_MM)))
    ep_cs = CartesianCoordinateSystem(ep_mesh)
    κ = ConstantCoefficient(SymmetricTensor{2,2,Float64}((KAPPA, 0.0, KAPPA)))
    model = MonodomainModel(
        ConstantCoefficient(1.0), ConstantCoefficient(1.0), κ,
        NoStimulationProtocol(), ionic, :φₘ, :s,
    )
    odeform = semidiscretize(
        ReactionDiffusionSplit(model, ep_cs),
        FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())), ep_mesh,
    )
    n_total = Thunderbolt.solution_size(odeform)
    dh = odeform.functions[1].dh; n_nodes = ndofs(dh)
    u₀ = zeros(Float64, n_total)
    u0_cell = Thunderbolt.default_initial_state(ionic, nothing)
    ionic_dofs = odeform.solution_indices[2]
    s0 = reshape(view(u₀, ionic_dofs), (n_nodes, Thunderbolt.num_states(ionic)))
    for i in 1:Thunderbolt.num_states(ionic); s0[:, i] .= u0_cell[i]; end

    ep_step = BackwardEulerSolver(;
        inner_solver=KrylovJL_CG(itmax=2000, atol=1e-8, rtol=1e-8, history=false),
        solution_vector_type=Vector{Float64},
        system_matrix_type=Thunderbolt.ThreadedSparseMatrixCSR{Float64, Int32},
    )
    cell_step = AdaptiveForwardEulerSubstepper(;
        solution_vector_type=Vector{Float64}, reaction_threshold=0.1, substeps=10,
    )
    ts = OS.LieTrotterGodunov((ep_step, cell_step))

    # Pre-pace
    pre_p = OS.OperatorSplittingProblem(odeform, copy(u₀), (0.0, prepace*BCL))
    pre_i = Thunderbolt.init(pre_p, ts; dt=DT₀)
    for _ in TimeChoiceIterator(pre_i, 0.0:BCL:(prepace-1)*BCL) end
    u_ss = copy(pre_i.u)

    # ECG setup
    κ_cpu = ConstantCoefficient(SymmetricTensor{2,2,Float64}((KAPPA, 0.0, KAPPA)))
    strategy = Thunderbolt.SequentialAssemblyStrategy(Thunderbolt.SequentialCPUDevice())
    diff_op = Thunderbolt.setup_assembled_operator(
        strategy,
        Thunderbolt.BilinearDiffusionIntegrator(κ_cpu, QuadratureRuleCollection(2), :φₘ),
        Thunderbolt.ThreadedSparseMatrixCSR{Float64, Int32}, dh,
    )
    Thunderbolt.update_operator!(diff_op, 0.0)
    ecg_cache = Thunderbolt.Plonsey1964ECGGaussCache(diff_op, zeros(n_nodes))
    electrode = Vec{2,Float64}((X_ELEC, Y_ELEC))

    # Measurement
    heat_dr = odeform.solution_indices[1]
    meas_p = OS.OperatorSplittingProblem(odeform, u_ss, (0.0, meas*BCL))
    meas_i = Thunderbolt.init(meas_p, ts; dt=DT₀)
    tr = 0.0:DT_VIS:(meas*BCL - DT_VIS)
    ecg_v = zeros(length(tr))
    idx = 0
    for (u_, _) in TimeChoiceIterator(meas_i, tr)
        idx += 1; idx > length(tr) && break
        φ = Float64.(u_[heat_dr])
        Thunderbolt.update_ecg!(ecg_cache, φ)
        ecg_v[idx] = Thunderbolt.evaluate_ecg(ecg_cache, electrode, 1.0)
    end

    # QT detection
    QTs = Float64[]
    step = round(Int, BCL/DT_VIS)
    for b in 0:(meas-1)
        i0=b*step+1; i1=min((b+1)*step, length(ecg_v))
        i1-i0 < 20 && continue
        beat=ecg_v[i0:i1]; n=length(beat); w=max(1,round(Int,0.05n))
        bl=0.5*(mean(beat[1:w])+mean(beat[end-w+1:end]))
        qe=round(Int,0.4n); qe<2 && continue
        qd=abs.(beat[1:qe].-bl); isempty(qd) && continue
        qpk=argmax(qd); qs=sign(beat[qpk]-bl); qi=1
        qt=bl+0.2*qs*qd[qpk]
        for i in 1:qpk; if (qs>0 ? beat[i]>qt : beat[i]<qt); qi=i; break; end; end
        ts2=round(Int,0.35n); te0=round(Int,0.95n); ts2>=te0 && continue
        tseg=beat[ts2:te0]; isempty(tseg) && continue
        td=abs.(tseg.-bl); tp=argmax(td)+ts2-1; tsgn=sign(beat[tp]-bl)
        tpk2=td[argmax(td)]; tt=bl+0.05*tsgn*tpk2; tei=te0
        for i in te0:-1:tp; i>length(beat)&&continue; if (tsgn>0 ? beat[i]>tt : beat[i]<tt); tei=i; break; end; end
        qt_ms=(tei-qi)*DT_VIS; 50.0<qt_ms<450.0 && push!(QTs,qt_ms)
    end

    n_qt = length(QTs)
    μ = n_qt >= 5 ? mean(QTs) : NaN
    σ = n_qt >= 5 ? std(QTs) : NaN
    qtvi = n_qt >= 5 ? log10(max(var(QTs),1e-8)/μ^2) - log10((1e-3/BCL)^2) : NaN
    @printf("    QT=%.1f±%.2f ms  QTVi=%.3f  (n=%d)\n", μ, σ, qtvi, n_qt)
    return QTs, collect(tr), ecg_v
end

println("="^60)
println("Quick 2D Tissue ($(N_EL)×$(N_EL) mesh, CPU Float64)")
println("="^60)

out = joinpath(@__DIR__, "..", "figures", "data")
mkpath(out)

# τ_f sweep
println("\n=== τ_f sweep ===")
tauf_vals = [30.0, 40.0, 45.0, 48.0, 52.0]
ecg_traces = Dict()
open(joinpath(out, "tissue2d_quick_tauf_qtv.csv"), "w") do io
    println(io, "tauf,qt_mean,qt_std,qtvi,n")
    for tf in tauf_vals
        t0=time()
        qts, et, ev = run_sim(tf, 3.0)
        println("  time: $(round(time()-t0,digits=1))s")
        n=length(qts)
        μ=n>=5?mean(qts):NaN; σ=n>=5?std(qts):NaN
        qtvi=n>=5 ? log10(max(var(qts),1e-8)/μ^2)-log10((1e-3/BCL)^2) : NaN
        println(io, "$tf,$μ,$σ,$qtvi,$n"); flush(io)
        tf in [30.0, 45.0, 52.0] && (ecg_traces["tauf$(round(Int,tf))"] = (et, ev))
    end
end
println("τ_f sweep done → tissue2d_quick_tauf_qtv.csv")

# u sweep
println("\n=== u (Ca instability) sweep ===")
u_vals = [3.0, 7.0, 8.5, 9.5]
open(joinpath(out, "tissue2d_quick_u_qtv.csv"), "w") do io
    println(io, "u,qt_mean,qt_std,qtvi,n")
    for uv in u_vals
        t0=time()
        qts, et, ev = run_sim(35.0, uv)
        println("  time: $(round(time()-t0,digits=1))s")
        n=length(qts)
        μ=n>=5?mean(qts):NaN; σ=n>=5?std(qts):NaN
        qtvi=n>=5 ? log10(max(var(qts),1e-8)/μ^2)-log10((1e-3/BCL)^2) : NaN
        println(io, "$uv,$μ,$σ,$qtvi,$n"); flush(io)
        uv in [3.0, 8.5] && (ecg_traces["u$(uv)"] = (et, ev))
    end
end
println("u sweep done → tissue2d_quick_u_qtv.csv")

# Save ECG traces
if !isempty(ecg_traces)
    ks = sort(collect(keys(ecg_traces)))
    open(joinpath(out, "tissue2d_ecg_traces.csv"), "w") do io
        println(io, "t_ms," * join(ks, ","))
        n = minimum(length(v[1]) for v in values(ecg_traces))
        for i in 1:n
            t_str = string(round(ecg_traces[ks[1]][1][i], digits=2))
            vals = join([string(round(ecg_traces[k][2][i], digits=8)) for k in ks], ",")
            println(io, "$t_str,$vals")
        end
    end
    println("ECG traces → tissue2d_ecg_traces.csv")
end

println("\n✓ Quick 2D tissue sim complete")
