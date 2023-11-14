# quadrature rule for npt^d PTR using the full grid
# all weights are assumed to be unity
# it is an immutable array
"""
    PTR{d}(x::AbstractVector{T}) where {d,T}

Stores a `d` dimensional Cartesian product grid of `SVector{d,T}`.
Similar to `Iterators.product(ntuple(n->x, d)...)`.
Uses the same number of grid points per dimension.
"""
struct PTR{N,T,X} <: AbstractArray{Tuple{One,SVector{N,T}},N}
    x::X
    function PTR{N}(x::X) where {N,X<:AbstractVector}
        @assert N isa Integer
        @assert N >= 1
        @assert length(x) > 0
        new{N,eltype(x),X}(x)
    end
end

# Array interface
Base.size(p::PTR) = (l=length(p.x); ntuple(n->l, Val(ndims(p))))
function Base.getindex(p::PTR{N,T}, idx::Vararg{Int,N}) where {N,T}
    return One(), SVector{N,T}(ntuple(n -> p.x[idx[n]], Val(N)))
end

# iterator interface
Base.IteratorSize(::Type{<:PTR{N}}) where {N} = Base.HasShape{N}()
Base.size(p::PTR, _) = length(p.x)
# iteration with a Cartesian index state (extremely similar to Iterators.product)
function Base.iterate(p::PTR{N,T}) where {N,T}
    next = iterate(p.x)
    next === nothing && return nothing
    val   = ntuple(i->next[1],Val(N))
    state = ntuple(i->next,Val(N))
    return (One(), SVector{N,T}(val)), state
end

Base.isdone(::PTR, state) = all(isnothing, state)
piterate(_, ::Nothing...) = nothing
function piterate(iter, state)
    next = iterate(iter, state[2])
    if next === nothing
        return nothing
    else
        return (next[1],), (next,)
    end
end
function piterate(iter, state, states::Vararg{Tuple{A,B},N}) where {A,B,N}
    next = iterate(iter, state[2])
    if next === nothing
        restnext = piterate(iter, states...)
        restnext === nothing && return nothing
        _, nextstates = restnext
        next = iterate(iter)
        next === nothing && return nothing
        return (next[1], ntuple(i->nextstates[i][1],Val(N))...), (next, nextstates...)
    else
        return (next[1], ntuple(i->states[i][1],Val(N))...), (next, states...)
    end
end
function Base.iterate(p::PTR{N,T}, state) where {N,T}
    next = piterate(p.x, state...)
    next === nothing && return nothing
    return ((One(), SVector{N,T}(next[1])), next[2])
end

# rule interface
PTR(::Type{T}, ::Val{d}, npt) where {T,d} = PTR{d}(ptrpoints(T, npt))
function (rule::PTR)(f::F, B::Basis, buffer=nothing) where {F}
    arule = AffineQuad(rule, B)
    return quadsum(arule, f, arule.vol / length(rule), buffer)
end

function (r::MonkhorstPackRule{Nothing})(::Type{T}, v::Val{d}) where {T,d}
    return PTR(T, v, r.n₀)
end

function nextrule(p::PTR{d,T}, r::MonkhorstPackRule) where {d,T}
    return PTR(T, Val(d), length(p.x)+r.Δn)
end
