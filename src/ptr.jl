# quadrature rule for npt^d PTR using the full grid
# all weights are assumed to be unity
# it is an immutable array
# quadrature nodes and weights should be unitless
ptrpoints(npt) = range(0//1, length=npt, step=1//npt)
ptrpoints(T::Type, npt) = float(real(one(T)))*ptrpoints(npt)

"""
    PTR{d}(x::AbstractVector{T}) where {d,T}

Stores a `d` dimensional Cartesian product grid of `SVector{d,T}`.
Similar to `Iterators.product(ntuple(n->x, d)...)`.
Uses the same number of grid points per dimension.
"""
struct PTR{N,T,X,W} <: AbstractQuadratureRule
    x::X
    w::W
    function PTR{N}(x::X, w=step(x)^d) where {N,X<:AbstractRange}
        @assert N isa Integer
        @assert N >= 1
        @assert length(x) > 0
        p = ProductArray(ntuple(_->x, Val{N}()))
        new{N,eltype(x),typeof(p),typeof(w)}(p, w)
    end
end
PTR(::Type{T}, ::Val{d}, npt, w=convert(T, 1//npt)^d) where {T,d} = PTR{d}(ptrpoints(T, npt), w)

# indexing interface
Base.@propagate_inbounds Base.getindex(p::PTR, args...) = (p.w, getindex(p.x, args...))
Base.firstindex(p::PTR) = 1
Base.lastindex(p::PTR) = length(p)

# quadrature rule multi-threading interface
Base.IndexStyle(::Type{<:PTR}) = IndexCartesian()
Base.axes(p::PTR, args...) = axes(p.x, args...)
Base.nextind(p::PTR, i::CartesianIndex) = nextind(p.x, i)

# iterator interface
Base.IteratorSize(::Type{<:PTR{N}}) where {N} = Base.HasShape{N}()
Base.length(p::PTR) = length(p.x)
Base.size(p::PTR, args...) = size(p.x, args...)
Base.eltype(::Type{<:PTR{N,T,X,W}}) where {N,T,X,W} = Tuple{W,SVector{N,T}}
function Base.iterate(p::PTR, args...)
    next = Base.iterate(p.x, args...)
    next === nothing && return nothing
    x, state = next
    return (p.w, x), state
end

