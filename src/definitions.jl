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

@inline mymul(::One, x) = x
@inline mymul(w, x) = w*x

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
quadsum(rule, f, B) = sum(((w,x),) -> mymul(w, f(B*x)), rule)

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
            buffer[i] += mymul(w, f(B*x))
        end
    end
    return sum(buffer)
end

quadsum(rule, f, B, ::Nothing) = quadsum(rule, f, B)
quadsum(rule, f, B, buffer) = parquadsum(rule, f, B, buffer)



struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Vector{Y}
    x::Vector{X}
    max_batch::Int # maximum number of x to supply in parallel (defaults to typemax(Int))
    function BatchIntegrand{F,Y,X}(f!::F, y::Vector{Y}, x::Vector{X}, max_batch::Int) where {F,Y,X}
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{F,Y,X}(f!, y, x, max_batch)
    end
end

BatchIntegrand(f!::F, y::Vector{Y}, x::Vector{X}; max_batch::Integer=typemax(Int)) where {F,Y,X} =
    BatchIntegrand{F,Y,X}(f!, y, x, max_batch)

function quadsum(rule, f::BatchIntegrand, B)
    # unroll first batch iteration to get right types
    n = countevals(rule)
    m = min(n, f.max_batch)
    resize!(f.x, m); resize!(f.y, m)
    prev = next = iterate(rule)
    next === nothing && throw(ArgumentError("empty rule"))
    i = j = 0
    while next !== nothing && i < m
        (_,x), state = next
        f.x[i += 1] = B*x
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
                f.x[(i += 1) - j] = B*x
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

    return I
end

function fill_xbuf!(xbuf::Vector, B, rule, r, off, nchunks)
    Threads.@threads for (xrange, _) in chunks(r, nchunks)
        for i in xrange
            _, x = rule[r[i]]
            xbuf[r[i]-off] = B*x
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

function parquadsum(rule, f::BatchIntegrand, B, buffer::Vector)
    (nthreads = min(Threads.nthreads(), f.max_batch)) == 1 && rule(f, B, nothing)

    n = countevals(rule)
    l = m = min(n, f.max_batch)
    resize!(f.x, m); resize!(f.y, m)
    fill_xbuf!(f.x, B, rule, 1:m,0, nthreads)
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
        fill_xbuf!(f.x, B, rule, r,l, nthreads)
        f.f!(f.y, f.x)
        quad_buf!(buffer, f.y, rule, r,l, nthreads)
        I += sum(buffer)
        l += k
    end

    return I
end