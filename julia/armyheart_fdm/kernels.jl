@kernel function ∇²_1d!(d2u, u, t, κ, stim, dx, N)
    i = @index(Global)

    if 2 <= i <= N - 1 # Interior points
        d2x = (u[i + 1] - 2u[i] + u[i - 1]) / (dx^2)
        #pos = Thunderbolt.Vec((i - 1) * dx)
        d2u[i] = κ * d2x #+ stim(pos, t)
    elseif i == 1 # Left boundary (i=1)
        d2x = 2(u[i + 1] - u[i]) / (dx^2)
        #pos = Thunderbolt.Vec(0.0)
        d2u[i] = κ * d2x #+ stim(pos, t)
    elseif i == N # Right boundary (i=N)
        d2x = 2(u[i - 1] - u[i]) / (dx^2)
        #pos = Thunderbolt.Vec((N - 1) * dx)
        d2u[i] = κ * d2x #+ stim(pos, t)
    end
end

function (m::MonodomainFD{1})(du, u, p, t)
    # Get the grid size and dx
    N = m.grid.N[1]
    dx = only(m.grid.dx)
    backend = get_backend(u)
    ∇²_1d!(backend)(du, u, t, m.κ, m.I_stim, dx, N; ndrange=N)
    KernelAbstractions.synchronize(backend)
    return nothing
end

@inline idx2flat(i, j, Nx) = (j - 1) * Nx + i

@kernel function ∇²_2d!(d2u, u, t, κ, stim, dx, dy, Nx, Ny)
    i, j = @index(Global, NTuple)
    idx = (j - 1) * Nx + i  # Flat index

    # Calculate spatial position
    pos = Thunderbolt.Vec(((i - 1) * dx, (j - 1) * dy))
    I_stim = stim(pos, t)

    if 2 <= i <= Nx - 1 && 2 <= j <= Ny - 1  # Interior points
        d2x = (u[idx2flat(i+1,j,Nx)] - 2u[idx] + u[idx2flat(i-1,j,Nx)]) / (dx^2)
        d2y = (u[idx2flat(i,j+1,Nx)] - 2u[idx] + u[idx2flat(i,j-1,Nx)]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == 1 && 2 <= j <= Ny - 1  # Left boundary
        d2x = 2(u[idx2flat(i+1,j,Nx)] - u[idx]) / (dx^2)
        d2y = (u[idx2flat(i,j+1,Nx)] - 2u[idx] + u[idx2flat(i,j-1,Nx)]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == Nx && 2 <= j <= Ny - 1  # Right boundary
        d2x = 2(u[idx2flat(i-1,j,Nx)] - u[idx]) / (dx^2)
        d2y = (u[idx2flat(i,j+1,Nx)] - 2u[idx] + u[idx2flat(i,j-1,Nx)]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif 2 <= i <= Nx - 1 && j == 1  # Bottom boundary
        d2x = (u[idx2flat(i+1,j,Nx)] - 2u[idx] + u[idx2flat(i-1,j,Nx)]) / (dx^2)
        d2y = 2(u[idx2flat(i,j+1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif 2 <= i <= Nx - 1 && j == Ny  # Top boundary
        d2x = (u[idx2flat(i+1,j,Nx)] - 2u[idx] + u[idx2flat(i-1,j,Nx)]) / (dx^2)
        d2y = 2(u[idx2flat(i,j-1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == 1 && j == 1  # Bottom-left corner
        d2x = 2(u[idx2flat(i+1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j+1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == Nx && j == 1  # Bottom-right corner
        d2x = 2(u[idx2flat(i-1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j+1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == 1 && j == Ny  # Top-left corner
        d2x = 2(u[idx2flat(i+1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j-1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    elseif i == Nx && j == Ny  # Top-right corner
        d2x = 2(u[idx2flat(i-1,j,Nx)] - u[idx]) / (dx^2)
        d2y = 2(u[idx2flat(i,j-1,Nx)] - u[idx]) / (dy^2)
        d2u[idx] = κ * (d2x + d2y) + I_stim
    end
end

function (m::MonodomainFD{2})(du, u, p, t)
    Nx, Ny = m.grid.N
    dx, dy = m.grid.dx
    backend = get_backend(u)
    ∇²_2d!(backend)(du, u, t, m.κ, m.I_stim, dx, dy, Nx, Ny; ndrange=(Nx, Ny))
    KernelAbstractions.synchronize(backend)
    return nothing
end
