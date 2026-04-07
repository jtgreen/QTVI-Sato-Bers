struct LazyGrid{Dim,TN,TL,TX}
    N::TN
    L::TL
    dx::TX
end
function LazyGrid(L, dx::T) where {T<:Real}
    N = round.(Int, L ./ dx) .+ 1
    if !any(((N .- 1) .* dx) .≈ L)
        @warn "LazyGrid size is not an integer multiple of dx. Adjusting L to fit. Old L ($L), New L: $(N.*dx)"
    end
    L = (N .- 1) .* dx
    _dx = ntuple(_ -> dx, length(L))
    return LazyGrid{length(L),typeof(N),typeof(L),typeof(_dx)}(N, L, _dx)
end
function LazyGrid(L, dx)
    @assert length(L) == length(dx) "L and dx must have the same length"
    N = round.(Int, L ./ dx) .+ 1
    if !any(((N .- 1) .* dx) .≈ L)
        @warn "LazyGrid size is not an integer multiple of dx. Adjusting L to fit. Old L ($L), New L: $(N.*dx)"
    end
    L = (N .- 1) .* dx
    return LazyGrid{length(L),typeof(N),typeof(L),typeof(dx)}(N, L, dx)
end

# Constructor for type conversion
function LazyGrid(grid::LazyGrid, ::Type{T}) where {T<:Real}
    N = grid.N  # Keep as integers
    L = T.(grid.L)  # Convert to new type
    dx = T.(grid.dx)  # Convert to new type
    return LazyGrid{length(L),typeof(N),typeof(L),typeof(dx)}(N, L, dx)
end
