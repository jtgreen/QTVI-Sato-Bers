using CUDA
using KernelAbstractions
using OhMyThreads
using ProgressMeter
using OrdinaryDiffEq
using OrdinaryDiffEqOperatorSplitting
using LinearAlgebra
using SciMLBase
using LockstepODE
import Thunderbolt: AbstractEPModel, num_states
import OrdinaryDiffEqOperatorSplitting as OS

struct MonodomainFD{Dim,G,T<:Real,F<:Function} <: AbstractEPModel
    grid::G
    κ::T
    I_stim::F
end
function MonodomainFD(grid::LazyGrid{Dim}, κ::T) where {Dim,T<:Real}
    no_stim(x, t) = 0
    return MonodomainFD{Dim,typeof(grid),T,typeof(no_stim)}(grid, κ, no_stim)
end

function MonodomainFD(grid::LazyGrid{Dim}, κ::T, stim::F) where {Dim,T<:Real,F}
    return MonodomainFD{Dim,typeof(grid),T,F}(grid, κ, stim)
end

# Constructor for type conversion
function MonodomainFD(model::MonodomainFD{Dim}, ::Type{T}) where {Dim,T<:Real}
    grid = LazyGrid(model.grid, T)
    κ = T(model.κ)
    return MonodomainFD{Dim,typeof(grid),T,typeof(model.I_stim)}(grid, κ, model.I_stim)
end

# ==================== Function Wrappers for Operator Splitting ====================

"""
    make_cell_ode(cellmodel)

Create cell ODE function that extracts spatial position from parameters.
Parameters should be a vector of NamedTuples: [(cellmodel=..., x_position=...), ...]
"""
function make_cell_ode(cellmodel)
    function cell_ode!(du, u, p, t)
        Thunderbolt.cell_rhs!(du, u, p.x_position, t, cellmodel)
    end
    return cell_ode!
end

"""
    FDMSignalingFunction

Wrapper for signaling dynamics that provides an ODE function interface.
"""
struct FDMSignalingFunction{SM,CT}
    signaling_model::SM
    constants::CT
end

function (f::FDMSignalingFunction)(du, u, p, t)
    # Call the signaling model's rhs function
    f.signaling_model(du, u, p, t)
    return nothing
end

num_states(f::FDMSignalingFunction) = Thunderbolt.num_states(f.signaling_model)

# ==================== Synchronization Objects ====================

"""
    VoltageExtractSync

Extracts voltage from interleaved cell states for diffusion operator.
Cell states are organized as [V1, s2_1, ..., sM_1, V2, s2_2, ..., sM_2, ...]
Voltage is at stride N_cell: indices 1, N_cell+1, 2*N_cell+1, ...
"""
struct VoltageExtractSync{N}
    N_cell::N  # Number of states per spatial point (stride)
end

function OS.forward_sync_external!(
    outer_integrator::OS.OperatorSplittingIntegrator,
    _inner_integrator::SciMLBase.DEIntegrator,
    sync::VoltageExtractSync,
)
    # No-op: voltage sync happens via shared DOF indices
    # The operators work on the same underlying array
    return nothing
end

"""
    SignalToCellTransferFDM

Synchronizes signaling pathway outputs to cell model parameters.
"""
struct SignalToCellTransferFDM{SM,PT,R}
    signaling_model::SM
    cell_parameters::PT  # View into cell model's CONSTANTS
    signal_dofrange::R   # DOF range for signaling states in the full state vector
end

function OS.forward_sync_external!(
    outer_integrator::OS.OperatorSplittingIntegrator,
    _inner_integrator::SciMLBase.DEIntegrator,
    syncer::SignalToCellTransferFDM,
)
    # Extract signaling state from the full state vector
    signaling_state = @view outer_integrator.u[syncer.signal_dofrange]
    signaling_constants = syncer.signaling_model.c

    # Update effective fractions
    CUDA.@allowscalar SignalingModel.effective_fractions!(
        syncer.cell_parameters, signaling_state, signaling_constants
    )

    # Calculate Whole_cell_PP1
    CUDA.@allowscalar pp1_PP1f_cyt_sum =
        signaling_constants[38] - signaling_constants[37] + signaling_state[39]
    CUDA.@allowscalar PP1f_cyt =
        0.5 * (
            √(pp1_PP1f_cyt_sum^2 + 4 * signaling_constants[38] * signaling_constants[37]) -
            pp1_PP1f_cyt_sum
        )
    CUDA.@allowscalar Whole_cell_PP1 = (
        signaling_constants[36] / signaling_constants[5] +
        signaling_constants[35] / signaling_constants[6] +
        PP1f_cyt / signaling_constants[7]
    )
    CUDA.@allowscalar syncer.cell_parameters[end] = Whole_cell_PP1

    return nothing
end

# ==================== Semidiscretization ====================

function semidiscretize(
    model::MonodomainFD{1},
    cellmodel,
    iso_conc,
    u0::AbstractVector{T};
    internal_threading=true,
) where {T<:AbstractFloat}
    backend = get_backend(u0)
    VectorType = backend isa CUDA.CUDABackend ? CuVector : Vector

    L = T(model.grid.L[1])
    dx = T(model.grid.dx[1])

    # Create spatial positions
    x = VectorType(Thunderbolt.Vec.(zero(T):dx:L))
    N_spatial = length(x)

    # Extract state components
    N_cell = num_states(cellmodel)
    N_signaling = 57
    total_cell_states = N_cell * N_spatial

    # Create signaling model
    signaling_model = GongBetaAdrenergicSignalingModel(VectorType{T}; iso_conc=iso_conc)

    # Get fraction parameter indices
    fraction_indices = [
        ToRORdTrauma.parameter_index("fICaLP"),
        ToRORdTrauma.parameter_index("fIKsP"),
        ToRORdTrauma.parameter_index("fPLBP"),
        ToRORdTrauma.parameter_index("fTnIP"),
        ToRORdTrauma.parameter_index("fINaP"),
        ToRORdTrauma.parameter_index("fINaKP"),
        ToRORdTrauma.parameter_index("fRyRP"),
        ToRORdTrauma.parameter_index("fIKurP"),
        ToRORdTrauma.parameter_index("Whole_cell_PP1"),
    ]
    cell_parameters = @view cellmodel.CONSTANTS[fraction_indices]

    # Create function wrappers for operator splitting
    # State layout: [V1, s2_1, ..., sM_1, V2, s2_2, ..., sM_2, ..., signaling...]
    # Voltage is at strided indices: 1, N_cell+1, 2*N_cell+1, ...
    voltage_indices = 1:N_cell:total_cell_states

    diffusion_fun = ODEFunction(model)

    # Create LockstepODE function for cell dynamics
    cell_ode = make_cell_ode(cellmodel)
    cell_params = [(cellmodel=cellmodel, x_position=x[i]) for i in 1:N_spatial]
    lockstep_func = LockstepFunction(
        cell_ode, N_cell, N_spatial; internal_threading=internal_threading
    )
    cell_fun = ODEFunction((du, u, p, t) -> lockstep_func(du, u, cell_params, t))

    signal_fun = ODEFunction(signaling_model)

    # Define DOF ranges
    # Interleaved layout: [V1, s2_1, ..., sM_1, V2, s2_2, ..., sM_2, ..., signaling...]
    # Voltage: every N_cell-th element (strided)
    # Cells: all interleaved states
    # Signaling: after cell states
    diffusion_dofrange = voltage_indices  # Strided voltage
    cell_dofrange = 1:total_cell_states   # All cell states (includes voltage)
    signal_dofrange = (total_cell_states + 1):(total_cell_states + N_signaling)

    # Create synchronization objects
    voltage_sync = VoltageExtractSync(N_cell)
    signal_to_cell_sync = SignalToCellTransferFDM(
        signaling_model, cell_parameters, signal_dofrange
    )

    # Create GenericSplitFunction
    semidiscrete = Thunderbolt.GenericSplitFunction(
        (diffusion_fun, cell_fun, signal_fun),
        (diffusion_dofrange, cell_dofrange, signal_dofrange),
        (voltage_sync, OS.NoExternalSynchronization(), signal_to_cell_sync),
    )

    return semidiscrete
end

function semidiscretize(
    model::MonodomainFD{2},
    cellmodel,
    iso_conc,
    u0::AbstractVector{T};
    internal_threading=true,
) where {T<:AbstractFloat}
    backend = get_backend(u0)
    VectorType = backend isa CUDA.CUDABackend ? CuVector : Vector

    dx, dy = T.(model.grid.dx)

    # Create 2D grid of spatial coordinates
    Nx, Ny = model.grid.N
    N_spatial = prod(model.grid.N)
    x_coords_temp = Vector{Thunderbolt.Vec{2,T}}(undef, N_spatial)
    for j in 1:Ny
        for i in 1:Nx
            idx = (j - 1) * Nx + i
            x_coords_temp[idx] = Thunderbolt.Vec((i - 1) * dx, (j - 1) * dy)
        end
    end
    x_coords = VectorType(x_coords_temp)

    # Extract state components
    N_cell = num_states(cellmodel)
    N_signaling = 57
    total_cell_states = N_cell * N_spatial

    # Create signaling model
    signaling_model = GongBetaAdrenergicSignalingModel(VectorType{T}; iso_conc=iso_conc)

    # Get fraction parameter indices
    fraction_indices = [
        ToRORdTrauma.parameter_index("fICaLP"),
        ToRORdTrauma.parameter_index("fIKsP"),
        ToRORdTrauma.parameter_index("fPLBP"),
        ToRORdTrauma.parameter_index("fTnIP"),
        ToRORdTrauma.parameter_index("fINaP"),
        ToRORdTrauma.parameter_index("fINaKP"),
        ToRORdTrauma.parameter_index("fRyRP"),
        ToRORdTrauma.parameter_index("fIKurP"),
        ToRORdTrauma.parameter_index("Whole_cell_PP1"),
    ]
    cell_parameters = @view cellmodel.CONSTANTS[fraction_indices]

    # Create function wrappers for operator splitting
    # State layout: [V1, s2_1, ..., sM_1, V2, s2_2, ..., sM_2, ..., signaling...]
    # Voltage is at strided indices: 1, N_cell+1, 2*N_cell+1, ...
    voltage_indices = 1:N_cell:total_cell_states

    diffusion_fun = SciMLBase.ODEFunction((du, u, p, t) -> model(du, u, p, t))

    # Create LockstepODE function for cell dynamics
    cell_ode = make_cell_ode(cellmodel)
    cell_params = [(cellmodel=cellmodel, x_position=x_coords[i]) for i in 1:N_spatial]
    lockstep_func = LockstepFunction(
        cell_ode, N_cell, N_spatial; internal_threading=internal_threading
    )
    cell_fun = SciMLBase.ODEFunction((du, u, p, t) -> lockstep_func(du, u, cell_params, t))

    signal_fun = SciMLBase.ODEFunction(
        (du, u, p, t) -> signaling_model(du, u, p, t); p=signaling_model.c
    )

    # Define DOF ranges
    # Interleaved layout: [V1, s2_1, ..., sM_1, V2, s2_2, ..., sM_2, ..., signaling...]
    # Voltage: every N_cell-th element (strided)
    # Cells: all interleaved states
    # Signaling: after cell states
    diffusion_dofrange = voltage_indices  # Strided voltage
    cell_dofrange = 1:total_cell_states   # All cell states (includes voltage)
    signal_dofrange = (total_cell_states + 1):(total_cell_states + N_signaling)

    # Create synchronization objects
    voltage_sync = VoltageExtractSync(N_cell)
    signal_to_cell_sync = SignalToCellTransferFDM(
        signaling_model, cell_parameters, signal_dofrange
    )

    # Create GenericSplitFunction
    semidiscrete = Thunderbolt.GenericSplitFunction(
        (diffusion_fun, cell_fun, signal_fun),
        (diffusion_dofrange, cell_dofrange, signal_dofrange),
        (voltage_sync, OS.NoExternalSynchronization(), signal_to_cell_sync),
    )

    return semidiscrete
end

function semidiscretize_rush_larsen(
    model::MonodomainFD{1},
    cellmodel,
    iso_conc,
    tspan,
    dt,
    u0::AbstractVector{T};
    internal_threading=true,
    diffusion_solver=Euler(),
    diffusion_dt=dt,
    backend=get_backend(u0),
    kwargs...,
) where {T<:AbstractFloat}
    VectorType = backend isa CUDA.CUDABackend ? CuVector : Vector

    L = T(model.grid.L[1])
    dx = T(model.grid.dx[1])

    tspan = T.(tspan)
    dt = T(dt)

    x = VectorType(Thunderbolt.Vec.(zero(T):dx:L))

    signaling_model = GongBetaAdrenergicSignalingModel(VectorType{T}; iso_conc=iso_conc)

    # Handle signaling state extraction
    N_signaling = 57  # Number of signaling ODEs
    total_cell_states = length(u0) - N_signaling

    u_signaling = copy(u0[(total_cell_states + 1):end])
    du_signaling = similar(u_signaling)
    u0_cells = u0[1:total_cell_states]

    # Get fraction parameter indices for signaling
    fraction_indices = [
        ToRORdTrauma.parameter_index("fICaLP"),
        ToRORdTrauma.parameter_index("fIKsP"),
        ToRORdTrauma.parameter_index("fPLBP"),
        ToRORdTrauma.parameter_index("fTnIP"),
        ToRORdTrauma.parameter_index("fINaP"),
        ToRORdTrauma.parameter_index("fINaKP"),
        ToRORdTrauma.parameter_index("fRyRP"),
        ToRORdTrauma.parameter_index("fIKurP"),
        ToRORdTrauma.parameter_index("Whole_cell_PP1"),
    ]

    # View into effective fractions in cell model constants
    effective_fractions = @view cellmodel.CONSTANTS[fraction_indices]

    # Determine if signaling pathway should run
    runSignalingPathway = iso_conc > 0

    # Create Rush-Larsen integrator for cell dynamics
    cell_integrator = RushLarsenIntegrator(
        cellmodel, u0_cells, dt, x, backend; internal_threading=internal_threading
    )
    V_cells = @view cell_integrator.u[1, :]

    # Create diffusion integrator
    diffusion_prob = OrdinaryDiffEq.ODEProblem(model, copy(V_cells), tspan)
    diffusion_integrator = OrdinaryDiffEq.init(
        diffusion_prob, diffusion_solver; dt=diffusion_dt, kwargs...
    )

    return (
        cell_integrator=cell_integrator,
        diffusion_integrator=diffusion_integrator,
        V_cells=V_cells,
        u_signaling=u_signaling,
        du_signaling=du_signaling,
        effective_fractions=effective_fractions,
        runSignalingPathway=runSignalingPathway,
        signaling_model=signaling_model,
        dt=dt,
        tspan=tspan,
    )
end

function semidiscretize_rush_larsen(
    model::MonodomainFD{2},
    cellmodel,
    iso_conc,
    tspan,
    dt,
    u0::AbstractVector{T};
    internal_threading=true,
    diffusion_solver=Euler(),
    diffusion_dt=dt,
    kwargs...,
) where {T<:AbstractFloat}
    backend = get_backend(u0)
    VectorType = backend isa CUDA.CUDABackend ? CuVector : Vector

    Lx, Ly = T.(model.grid.L)
    dx, dy = T.(model.grid.dx)

    tspan = T.(tspan)
    dt = T(dt)

    # Create 2D grid of spatial coordinates
    Nx, Ny = model.grid.N
    x_coords_temp = Vector{Thunderbolt.Vec{2,T}}(undef, prod(model.grid.N))
    for j in 1:Ny
        for i in 1:Nx
            idx = (j - 1) * Nx + i
            x_coords_temp[idx] = Thunderbolt.Vec((i - 1) * dx, (j - 1) * dy)
        end
    end
    x_coords = VectorType(x_coords_temp)

    signaling_model = GongBetaAdrenergicSignalingModel(VectorType{T}; iso_conc=iso_conc)

    # Handle signaling state extraction
    N_signaling = 57  # Number of signaling ODEs
    total_cell_states = length(u0) - N_signaling

    u_signaling = copy(u0[(total_cell_states + 1):end])
    du_signaling = similar(u_signaling)
    u0_cells = u0[1:total_cell_states]

    # Get fraction parameter indices for signaling
    fraction_indices = [
        ToRORdTrauma.parameter_index("fICaLP"),
        ToRORdTrauma.parameter_index("fIKsP"),
        ToRORdTrauma.parameter_index("fPLBP"),
        ToRORdTrauma.parameter_index("fTnIP"),
        ToRORdTrauma.parameter_index("fINaP"),
        ToRORdTrauma.parameter_index("fINaKP"),
        ToRORdTrauma.parameter_index("fRyRP"),
        ToRORdTrauma.parameter_index("fIKurP"),
        ToRORdTrauma.parameter_index("Whole_cell_PP1"),
    ]

    # View into effective fractions in cell model constants
    effective_fractions = @view cellmodel.CONSTANTS[fraction_indices]

    # Determine if signaling pathway should run
    runSignalingPathway = iso_conc > 0

    # Create Rush-Larsen integrator for cell dynamics
    cell_integrator = RushLarsenIntegrator(
        cellmodel, u0_cells, dt, x_coords, backend; internal_threading=internal_threading
    )
    V_cells = @view cell_integrator.u[1, :]

    # Create diffusion integrator
    diffusion_prob = OrdinaryDiffEq.ODEProblem(model, copy(V_cells), tspan)
    diffusion_integrator = OrdinaryDiffEq.init(
        diffusion_prob, diffusion_solver; dt=diffusion_dt, kwargs...
    )

    return (
        cell_integrator=cell_integrator,
        diffusion_integrator=diffusion_integrator,
        V_cells=V_cells,
        u_signaling=u_signaling,
        du_signaling=du_signaling,
        effective_fractions=effective_fractions,
        runSignalingPathway=runSignalingPathway,
        signaling_model=signaling_model,
        dt=dt,
        tspan=tspan,
    )
end

function step_monodomain!(setup)
    # Step cells using Rush-Larsen
    step!(setup.cell_integrator, setup.dt, true)

    # Sync voltage to diffusion
    @. setup.diffusion_integrator.u = setup.V_cells

    # Advance diffusion to match cell time
    target_time = setup.cell_integrator.t
    if setup.diffusion_integrator.t < target_time
        OrdinaryDiffEq.step!(
            setup.diffusion_integrator, target_time - setup.diffusion_integrator.t, true
        )
    end

    # Write back diffused voltage to cells
    @. setup.V_cells = setup.diffusion_integrator.u
    return nothing
end

function OrdinaryDiffEq.solve(setup; verbose=false)
    # Main stepping loop
    t = setup.tspan[1]
    t_end = setup.tspan[2]
    p = verbose ? Progress(round(Int, (t_end - t) / setup.dt); dt=1.0) : nothing

    while t < t_end
        # Update signaling fractions
        CUDA.@allowscalar begin
            update_fraction_parameters!(
                setup.runSignalingPathway,
                setup.dt,
                setup.du_signaling,
                setup.u_signaling,
                setup.signaling_model.c,
                setup.effective_fractions,
            )
        end

        step_monodomain!(setup)

        t = setup.cell_integrator.t
        verbose && next!(p)
    end

    sol = setup.diffusion_integrator.sol
    return sol.t, sol.u
end
