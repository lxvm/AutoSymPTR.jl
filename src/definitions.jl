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

# a trivial quadrature weight for PTR
struct One end

Base.:*(::One, x) = x

# utilities for defining and incrementing the number of ptr points
nextnpt(a, nmin, nmax, Δn) = min(max(nmin, round(Int, Δn/a)), nmax)
ptrpoints(npt) = float(range(0, length=npt, step=1//npt))
# map a linear index to a Cartesian index
function ptrindex(npt::Int, ::Val{N}, i::Int) where {N}
    pow = cumprod(ntuple(n->n==1 ? 1 : npt, Val(N)))
    pow[1] <= i <= npt*pow[N] || throw(BoundsError(p, i))
    idx = ntuple(n -> rem(div(i-1, pow[n]), npt)+1, Val(N))
    return CartesianIndex(idx)
end
# map a Cartesian index to a linear index
function ptrindex(npt::Int, i::CartesianIndex)
    idx = 0
    for j in reverse(i.I)
        1 <= j <= npt || throw(BoundsError(p, i.I))
        idx *= npt
        idx += j-1
    end
    return idx+1
end



struct MonkhorstPackRule{S}
    syms::S     # collection of point group symmetries
    n₀::Int64   # the initial number of grid points to use
    Δn::Int64   # the number of grid points to add with each increment
    MonkhorstPackRule{S}(syms::S, n₀::Int64, Δn::Int64) where {S} = new{S}(syms, n₀, Δn)
end

# we choose the initial guess to be 6/a and the increment log(10)/a
function MonkhorstPackRule(syms::S, a::Real, nmin::Integer=50, nmax::Integer=1000, n₀=6.0, Δn=log(10)) where {S}
    n0 = nextnpt(a, nmin, nmax, n₀)
    dn = nextnpt(a, nmin, nmax, Δn)
    return MonkhorstPackRule{S}(syms, n0, dn)
end

nsyms(rule::MonkhorstPackRule) = isnothing(rule.syms) ? 1 : length(rule.syms)


# we expect rules to be iterable and indexable and return (w, x) = rule[i]
quadsum(rule, f, B) = sum(((w,x),) -> w*f(B*x), rule)

function parquadsum(rule, f, B, buffer::Vector)
    (nthreads = Threads.nthreads()) == 1 && rule(f, B, nothing)
    n = countevals(rule)
    d, r = divrem(n, nthreads)
    resize!(buffer, nthreads)
    fill!(buffer, zero(eltype(buffer)))
    Threads.@threads for i in Base.OneTo(nthreads)
        # batch nodes into `nthreads` continguous groups of size d or d+1 (remainder)
        jmax = (i <= r ? d+1 : d)
        offset = min(i-1, r)*(d+1) + max(i-1-r, 0)*d
        @inbounds for j in 1:jmax
            w, x = rule[offset + j]
            buffer[i] += w*f(B*x)
        end
    end
    return sum(buffer)
end

quadsum(rule, f, B, ::Nothing) = quadsum(rule, f, B)
quadsum(rule, f, B, buffer) = parquadsum(rule, f, B, buffer)
