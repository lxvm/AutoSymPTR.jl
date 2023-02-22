# quadrature rule for npt^d PTR using the full grid

"""
    PTRGrid{d}(x::Vector{T}) where {d,T}

Stores a `d` dimensional Cartesian product grid of `SVector{d,T}`.
Similar to `Iterators.product(ntuple(n->x, d)...)`.
Uses the same number of grid points per dimension.
"""
struct PTRGrid{N,T}
    x::Vector{T}
    function PTRGrid{N}(x::Vector{T}) where {N,T}
        @assert N isa Integer
        @assert N >= 1
        new{N,T}(x)
    end
end

Base.ndims(::PTRGrid{N,T}) where {N,T} = N
Base.eltype(::Type{PTRGrid{N,T}}) where {N,T} = SVector{N,T}
Base.length(p::PTRGrid) = length(p.x)^ndims(p)
Base.size(p::PTRGrid) = (l=length(p.x); ntuple(n->l, Val(ndims(p))))
Base.copy!(p::PTRGrid, v::AbstractVector) = copy!(p.x, v)
Base.copy!(p::PTRGrid, q::PTRGrid) = copy!(p, q.x)

# map a linear index to a Cartesian index
function ptrindex(p::PTRGrid, i::Int)
    npt = length(p.x)
    pow = cumprod(ntuple(n->n==1 ? 1 : npt, Val(ndims(p))))
    pow[1] <= i <= npt*pow[ndims(p)] || throw(BoundsError(p, i))
    CartesianIndex(ntuple(n -> rem(div(i-1, pow[n]), npt)+1, Val(ndims(p))))
end

# map a Cartesian index to a linear index
function ptrindex(p::PTRGrid{N}, i::CartesianIndex{N}) where N
    npt = length(p.x)
    idx = 0
    for j in reverse(i.I)
        1 <= j <= npt || throw(BoundsError(p, i.I))
        idx *= npt
        idx += j-1
    end
    idx+1
end

function Base.getindex(p::PTRGrid{N,T}, i::Int) where {N,T}
    idx = ptrindex(p, i)
    SVector{N,T}(ntuple(n -> p.x[idx[n]], Val(N)))
end
Base.getindex(p::PTRGrid{N,T}, idx::CartesianIndex{N}) where {N,T} =
    SVector{N,T}(ntuple(n -> p.x[idx[n]], Val(N)))

Base.isdone(p::PTRGrid{N,T}, state) where {N,T} = !(1 <= state <= length(p.x)^N)
function Base.iterate(p::PTRGrid{N,T}, state=1) where {N,T}
    Base.isdone(p, state) && return nothing
    (p[state], state+1)
end

# nodes, with all weights assumed to be unity
struct PTRRule{N,T}
    x::PTRGrid{N,T}
end
Base.length(r::PTRRule) = length(r.x)
function Base.copy!(r::T, s::T) where {T<:PTRRule}
    copy!(r.x, s.x)
    r
end

function ptr_rule(::Type{T}, ::Val{d}) where {T,d}
    x = Vector{T}(undef, 0)
    PTRRule(PTRGrid{d}(x))
end

"""
    ptr_rule(npt, ::Val{d}, ::Type{T}) where {d,T}

Returns the grid points and weights, `x,w`, to use for an `npt` PTR quadrature
of `f` on the unit cube `d` and type `T`.
"""
ptr_rule(::Type{T}, npt, ::Val{d}) where {T,d} =
    ptr_rule!(ptr_rule(T, Val(d)), npt, Val(d))

function ptr_rule!(rule::PTRRule, npt, ::Val{d}) where d
    copy!(rule.x, range(0, 1, length=npt+1)[1:npt])
    rule
end

"""
    ptr(f, B::AbstractMatrix, syms; npt=npt_update(f,0), rule=ptr_rule(,f,B))

Evaluates the `npt^d` point PTR `rule` on the integrand `f`. The coordinates are
mapped into the basis `B`, whose basis vectors are stored as columns of the
matrix. The integral is returned.
"""
function ptr(f, B::AbstractMatrix; npt=npt_update(f, 0), rule=nothing)
    d = checksquare(B); T = float(eltype(B))
    rule_ = (rule===nothing) ? ptr_rule(T, npt, Val(d)) : rule
    int = sum(x -> f(B*x), rule_.x)
    int * det(B)/npt^d
end
