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
    AffineQuad(rule, A, b, c, vol)

A wrapper to an iterable and indexable quadrature rule that applies an affine coordinate
transformation to the nodes of the form `A*(x+c)+b`. While the weights should be rescaled by the
Jacobian determinant `vol=abs(det(A))`, the value is stored in the `vol` field of the struct
so that the caller can minimize multiplication by applying the rescaling after computing the
rule.

While the quadrature rule should be unitless, the affine transform may have units.
"""
struct AffineQuad{TR,TA,Tb,Tc,TV}
    rule::TR
    A::TA
    b::Tb
    c::Tc
    vol::TV
end

apply_affine(aq::AffineQuad, x) =  myadd(mymul(aq.A, myadd(x, aq.c)), aq.b)

function Base.iterate(aq::AffineQuad, args...)
    next = iterate(aq.rule, args...)
    next === nothing && return nothing
    (w, x), state = next
    return (w, apply_affine(aq, x)), state
end
Base.isdone(aq::AffineQuad, args...) = Base.isdone(aq.rule, args...)
Base.IteratorSize(::Type{<:AffineQuad{TR}}) where {TR} = Base.IteratorSize(TR)
Base.length(aq::AffineQuad) = length(aq.rule)
Base.size(aq::AffineQuad) = size(aq.rule)
# the eltype could change due to the affine transform
Base.IteratorEltype(::Type{<:AffineQuad{TR}}) where {TR} = Base.IteratorEltype(TR)
Base.eltype(::Type{<:AffineQuad{TR}}) where {TR} = eltype(TR)

function Base.getindex(aq::AffineQuad, i)
    w, x = getindex(aq.rule, i)
    return w, apply_affine(aq, x)
end
Base.firstindex(aq::AffineQuad) = firstindex(aq.rule)
Base.lastindex(aq::AffineQuad)  = lastindex(aq.rule)

AffineQuad(rule) = AffineQuad(rule, One(), Zero(), Zero(), 1)
AffineQuad(rule, B::Basis) = AffineQuad(rule, B.B, Zero(), Zero(), abs(det(B.B)))

# utilities for defining and incrementing the number of ptr points
nextnpt(a, nmin, nmax, Δn) = min(max(nmin, round(Int, Δn/a)), nmax)
# quadrature nodes and weights should be unitless
ptrpoints(npt) = range(0//1, length=npt, step=1//npt)
ptrpoints(T::Type, npt) = float(real(one(T)))*ptrpoints(npt)
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
quadsum(rule, f::F, vol) where {F} = vol * sum(((w,x),) -> mymul(w, f(x)), rule)

quadsum(rule, f::F, vol, buffer) where {F} = quadsum(rule, f, vol)


struct InplaceIntegrand{F,T<:AbstractArray,Y<:AbstractArray}
    # in-place function f!(y, x) that takes one x value and outputs an array of results in-place
    f!::F
    I::T
    Itmp::T
    y::Y
    ytmp::Y
end

"""
    InplaceIntegrand(f!, result::AbstractArray)

Constructor for a `InplaceIntegrand` accepting an integrand of the form `f!(y,x)`. The
caller also provides an output array needed to store the result of the quadrature.
Intermediate `y` arrays are allocated during the calculation, and the final result is
may or may not be written to `result`, so use the IntegralSolution immediately after the
calculation to read the result, and don't expect it to persist if the same integrand is used
for another calculation.
"""
function InplaceIntegrand(f!, result)
    return InplaceIntegrand(f!, result, similar(result), similar(result, Nothing), similar(result, Nothing))
end

function quadsum(rule, f::InplaceIntegrand, vol)
    next = iterate(rule)
    next === nothing && throw(ArgumentError("empty rule"))
    (w, x), state = next
    f.f!(f.y, x)
    I = f.ytmp .= mymul.(w, f.y)
    next = iterate(rule, state)
    while next !== nothing
        (w, x), state = next
        f.f!(f.y, x)
        I .+= mymul.(w, f.y)
        next = iterate(rule, state)
    end
    return f.I .= I .* vol
end

function quadsum(rule, f::InplaceIntegrand{F,<:AbstractArray,<:AbstractArray{Nothing}}, vol) where {F}
    y = f.I/vol
    g = InplaceIntegrand(f.f!, f.I, f.Itmp, y, similar(y))
    return quadsum(rule, g, vol)
end

struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Y
    x::X
    max_batch::Int # maximum number of x to supply in parallel (defaults to typemax(Int))
    function BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch::Integer=typemax(Int))
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{typeof(f!),typeof(y),typeof(x)}(f!, y, x, max_batch)
    end
end

BatchIntegrand(f!, y, x; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, y, x, max_batch)

function quadsum(rule, f::BatchIntegrand, vol)
    # unroll first batch iteration to get right types
    n = length(rule)
    m = min(n, f.max_batch)
    resize!(f.x, m); resize!(f.y, m)
    prev = next = iterate(rule)
    next === nothing && throw(ArgumentError("empty rule"))
    i = j = 0
    while next !== nothing && i < m
        (_,x), state = next
        f.x[i += 1] = x
        next = iterate(rule, state)
    end
    f.f!(f.y, f.x)
    (w,_), state = prev
    I = mymul(w, f.y[j += 1])
    prev = iterate(rule, state)
    while prev !== nothing && j < m
        (w,_), state = prev
        I += mymul(w, f.y[j += 1])
        prev = iterate(rule, state)
    end
    # accumulate remainder
    while j < n
        if i == j
            while next !== nothing && i-j < m
                (_,x), state = next
                f.x[(i += 1) - j] = x
                next = iterate(rule, state)
            end
            if next === nothing
                resize!(f.x, i-j)
                resize!(f.y, i-j)
            end
            f.f!(f.y, f.x)
        else
            k = j
            while prev !== nothing && j < i
                (w,_), state = prev
                I += mymul(w, f.y[(j += 1) - k])
                prev = iterate(rule, state)
            end
        end
    end

    return I*vol
end

quadsum(rule, f::BatchIntegrand, vol, buffer::Vector) = parquadsum(rule, f, vol, buffer)

function fill_xbuf!(xbuf::Vector, rule, r, off, nchunks)
    Threads.@threads for (xrange, _) in chunks(r, nchunks)
        for i in xrange
            _, x = rule[r[i]]
            xbuf[r[i]-off] = x
        end
    end
end

function quad_buf!(buffer::Vector, fy::Vector, rule, r, off, nchunks)
    Threads.@threads for (wrange, ichunk) in chunks(r, nchunks)
        buffer[ichunk] = zero(eltype(buffer))
        for i in wrange
            w, _ = rule[r[i]]
            buffer[ichunk] += mymul(w, fy[r[i]-off])
        end
    end
end

# we parallelize the filling of the quadrature buffers, but the BatchIntegrand needs to
# parallelize the integrand evaluations
function parquadsum(rule, f::BatchIntegrand, vol, buffer::Vector)
    nthreads = (len=length(buffer)) == 0 ? Threads.nthreads() : min(Threads.nthreads(), len)
    nthreads == 1 && return quadsum(rule, f, vol)

    n = length(rule)
    l = m = min(n, f.max_batch)
    resize!(f.x, m); resize!(f.y, m)
    fill_xbuf!(f.x, rule, 1:m,0, nthreads)
    f.f!(f.y, f.x)
    resize!(buffer, min(m, nthreads))
    quad_buf!(buffer, f.y, rule, 1:m,0, nthreads)
    I = sum(buffer)

    # accumulate remainder
    while l < n
        k = min(n-l, f.max_batch)
        if k < length(f.x)
            resize!(f.x, k)
            resize!(f.y, k)
        end
        k < length(buffer) && resize!(buffer, k)
        r = l+1:l+k
        fill_xbuf!(f.x, rule, r,l, nthreads)
        f.f!(f.y, f.x)
        quad_buf!(buffer, f.y, rule, r,l, nthreads)
        I += sum(buffer)
        l += k
    end
    resize!(buffer, nthreads) # reset to original size

    return I*vol
end

# TODO support BatchIntegrand(InplaceIntegrand)
