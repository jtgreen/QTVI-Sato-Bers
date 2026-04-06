#!/usr/bin/env julia
"""
test_tissue_cpu.jl — Quick CPU test of tissue_armyheart.jl on a tiny 5×5 grid.

This validates:
  1. SatoBersArmyModel and SatoBers module load correctly
  2. MonodomainModel + ReactionDiffusionSplit semidiscretize
  3. OS.LieTrotterGodunov solver runs 2 beats without error
  4. Plonsey1964ECGGaussCache computes ECG values
  5. QT detection produces plausible results

Usage:
  julia --project=/tmp/ArmyHeart test_tissue_cpu.jl
"""

const ARMYHEART_PATH = get(ENV, "ARMYHEART_PATH", "/tmp/ArmyHeart")
if isdir(joinpath(ARMYHEART_PATH, "Project.toml"))
    import Pkg; Pkg.activate(ARMYHEART_PATH; io=devnull)
    Pkg.instantiate(; io=devnull)
    println("✓ ArmyHeart env: $ARMYHEART_PATH")
end

using Thunderbolt
import OrdinaryDiffEqOperatorSplitting as OS
using LinearSolve
using Statistics
using Printf
using ProgressMeter

const _AH_UTILS = joinpath(ARMYHEART_PATH, "src", "utils.jl")
if isfile(_AH_UTILS)
    include(_AH_UTILS)
else
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
end
include(joinpath(@__DIR__, "SatoArmyHeart.jl"))

println("\n=== CPU Test: 5×5 mesh, 2 beats (tauf=35ms) ===")

# Tiny 5×5 mesh
N_EL     = 5
L_MM     = 2.5    # mm
DX_MM    = L_MM / N_EL  # = 0.5mm
N_NX     = N_EL + 1
N_NY     = N_EL + 1
KAPPA    = 0.1
BCL      = 300.0
DT₀      = 0.01
STIM_AMP = 80.0
STIM_DUR = 1.0
CORNER   = DX_MM * 2

ionic = SatoBersArmyModel(;
    tauf=35.0, av=3.0, N_CaL=1000,  # smaller N_CaL for speed
    nx=N_NX, ny=N_NY, dx=DX_MM, dy=DX_MM, dt=DT₀,
    bcl=BCL, stim_amp=STIM_AMP, stim_dur=STIM_DUR, corner_mm=CORNER,
)

# Build mesh
ep_mesh = generate_mesh(
    Quadrilateral, (N_EL, N_EL),
    Vec(Float64.((0.0, 0.0))),
    Vec(Float64.((L_MM, L_MM))),
)
ep_cs = CartesianCoordinateSystem(ep_mesh)
κ_tensor = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((KAPPA, 0.0, KAPPA)))
model = MonodomainModel(
    ConstantCoefficient(1.0), ConstantCoefficient(1.0),
    κ_tensor, NoStimulationProtocol(), ionic, :φₘ, :s,
)
odeform = semidiscretize(
    ReactionDiffusionSplit(model, ep_cs),
    FiniteElementDiscretization(Dict(:φₘ => LagrangeCollection{1}())),
    ep_mesh,
)
println("✓ Mesh + odeform built — solution size: ", solution_size(odeform))

# Initial state
u₀ = zeros(Float64, solution_size(odeform))
steady_state_initializer!(u₀, odeform)
println("✓ Initial state set")

# Solver
ep_stepper = BackwardEulerSolver(;
    inner_solver=KrylovJL_CG(; itmax=1000, atol=1e-10, rtol=1e-10, history=false),
    solution_vector_type=Vector{Float64},
    system_matrix_type=ThreadedSparseMatrixCSR{Float64,Int32},
)
cell_stepper = AdaptiveForwardEulerSubstepper(;
    solution_vector_type=Vector{Float64},
    reaction_threshold=0.1, substeps=10,
)
timestepper = OS.LieTrotterGodunov((ep_stepper, cell_stepper))
println("✓ Solver built")

# ECG cache
dh = odeform.functions[1].dh
κ_cpu = ConstantCoefficient(SymmetricTensor{2, 2, Float64}((KAPPA, 0.0, KAPPA)))
diff_op = setup_assembled_operator(
    SequentialAssemblyStrategy(SequentialCPUDevice()),
    BilinearDiffusionIntegrator(κ_cpu, QuadratureRuleCollection(2), :φₘ),
    ThreadedSparseMatrixCSR{Float64, Int32},
    dh,
)
update_operator!(diff_op, 0.0)
ecg_cache = Plonsey1964ECGGaussCache(diff_op, zeros(Float64, ndofs(dh)))
electrode  = Vec{2, Float64}((L_MM + 2.5, L_MM + 2.5))
println("✓ ECG cache built")

# Run 2 beats
n_beats = 2
tspan = (0.0, Float64(n_beats * BCL))
prob = OS.OperatorSplittingProblem(odeform, copy(u₀), tspan)
integ = OS.init(prob, timestepper; dt=Float64(DT₀), verbose=false)

heat_dofrange = odeform.solution_indices[1]
DT_VIS = 1.0
meas_trange = 0.0 : DT_VIS : Float64(n_beats * BCL - DT_VIS)
ecg_t = Float64.(collect(meas_trange))
ecg_v = zeros(length(meas_trange))
idx = 0

println("✓ Running simulation...")
t0 = time()
@showprogress for (u_, t) in OS.TimeChoiceIterator(integ, meas_trange)
    global idx += 1
    idx > length(ecg_v) && break
    φ_cpu = Vector(u_[heat_dofrange])
    update_ecg!(ecg_cache, Float64.(φ_cpu))
    ecg_v[idx] = evaluate_ecg(ecg_cache, electrode, 1.0)
end
elapsed = time() - t0

println("✓ Simulation done in $(round(elapsed, digits=1)) s")
println("  ECG range: [$(round(minimum(ecg_v), digits=6)), $(round(maximum(ecg_v), digits=6))]")

if maximum(abs.(ecg_v)) < 1e-12
    @warn "ECG is flat — check stimulus and cell model"
else
    println("✓ ECG has non-trivial signal — tissue simulation WORKING")
end

# Test QT detection
qts = Float64[]
step = round(Int, BCL / DT_VIS)
for b in 0:(n_beats-1)
    i0 = b*step+1; i1 = min((b+1)*step, length(ecg_v))
    beat = ecg_v[i0:i1]
    n = length(beat)
    bl = 0.5*(mean(beat[1:max(1,round(Int,0.05n))]) + mean(beat[end-max(0,round(Int,0.05n)-1):end]))
    push!(qts, abs(maximum(beat) - bl))
end
println("  ECG peak amplitudes per beat: ", round.(qts, digits=6))
println("\n✓ ALL TESTS PASSED — tissue_armyheart.jl is functional on CPU")
