# This is the domain we use: basis vectors in each column
struct Basis{d,T,A}
    B::A
    function Basis(B::AbstractMatrix)
        return new{checksquare(B),eltype(B),typeof(B)}(B)
    end
end

Base.ndims(::Basis{d}) where {d} = d
Base.eltype(::Type{<:Basis{d,T}}) where {d,T} = T
Base.:*(b::Basis, A) = b.B*A

struct Zero end

@inline myadd(x, ::Zero) = x
@inline myadd(x, y) = x + y

# a trivial quadrature weight for PTR
struct One <: Number end

@inline mymul(::One, x) = x
@inline mymul(w, x) = w*x

"""
    AbstractQuadratureRule

Supertype for quadrature rules with the following interface:
- They are iterable collections of `(w, x)` pairs of weights and nodes
In order to enable multi-threaded evaluation of the quadrature rule, the objects must also
- Implement `firstindex`, `lastindex`, `nextind`, and `length` (`size` and `axes`) with linear (Cartesian) indexing
The `IndexStyle` of the rule is used to decide which type of indexing to use.
"""
abstract type AbstractQuadratureRule end

is_chunkable(::Q) where {Q<:AbstractQuadratureRule} = true
(rule::AbstractQuadratureRule)(f) = evalrule(rule, f)

# we expect rules to be iterable and indexable and return (w, x) = rule[i]
evalrule(rule::AbstractQuadratureRule, f) = _evalrule(rule, f)
_evalrule(rule, f::F) where {F} = sum(((w,x),) -> mymul(w, f(x)), rule)


# data type for product quadratures
struct ProductArray{T,N,X<:NTuple{N,<:AbstractVector{T}}} <: AbstractArray{T,N}
    xs::X
end
Base.size(p::ProductArray) = map(length, p.xs)
# Base.@propagate_inbounds function Base.getindex(p::ProductArray{T,2}, idx::Vararg{Int,2}) where {T}
#     SVector{2,T}((getindex(p.xs[1], idx[1]), getindex(p.xs[2], idx[2])))
# end
Base.@propagate_inbounds function Base.getindex(p::ProductArray{T,N}, idx::Vararg{Int,N}) where {T,N}
    items = _getindex(p.xs, idx...)
    SVector{N,T}(items)
end
# unroll map(getindex, p.xs, idx) since inbounds cannot propagate through a high-order function, even ntuple(f, ::Val)
_getindex(::Tuple{}) = ()
Base.@propagate_inbounds function _getindex(xs::Tuple{Vararg{AbstractVector, N}}, i::Vararg{Int,N}) where {N}
    return (_getindex(xs[1:N-1], i[1:N-1]...)..., getindex(xs[N], i[N]))
end

Base.@propagate_inbounds Base.getindex(p::ProductArray{T,N}, idx::CartesianIndex{N}) where {T,N} = getindex(p, Tuple(idx)...)

Base.IteratorSize(::Type{<:ProductArray{T,N}}) where {T,N} = Base.HasShape{N}()
Base.size(p::ProductArray, dim) = length(p.xs[dim])
Base.axes(p::ProductArray) = map(eachindex, p.xs)
