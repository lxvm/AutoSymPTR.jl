# quadrature rule for npt^d PTR reduced by a collection of symmetries
# computes the value of the integral over the IBZ

# nodes and weights
struct MonkhorstPack{d,W,X}
    npt::Int64
    nsyms::Int64
    wx::Vector{Tuple{W,SVector{d,X}}}
end

# indexing interface
Base.getindex(rule::MonkhorstPack, i::Int) = rule.wx[i]

# iteration interface
Base.eltype(::Type{MonkhorstPack{d,T}}) where {d,T} = Tuple{Int64,SVector{d,T}}
Base.iterate(rule::MonkhorstPack, args...) = iterate(rule.wx, args...)
Base.length(rule::MonkhorstPack) = length(rule.wx)

# TODO: parallelize this algorithm, which can be a bottleneck for large `npt`
# IDEA: nearby points are not related by symmetry except at high-symmetry planes, and so
# they can be parallelized in a thread-safe way. Since the IBZ is convex, as long as we
# chunk points contiguously so that they don't pass through a high-symmetry plane it is ok
# NEW IDEA: There is an O(N^(d-1)) algorithm that uses the convexity of the IBZ to employ
# line searching along the boundary of the polytope. Instead of putting flags everywhere on
# the wsym array, we can use the number of distinct symmetric images to get the quadrature
# weight and we can use the distance between a point and its images to determine whether it
# is along a boundary (i.e. has weight < length(syms) or is adjacent to one of its symmetric
# images)
# EMBARRASINGLY PARALLEL IDEA: use the symmetries to compute the H-representation of the IBZ
# and then test all k-points for inclusion in the polytope, with some edge cases for the
# weights
"""
    symptr_rule(npt, ::Val{d}, syms, T::Integer=UInt8) where {d}

Returns `wsym`, a `npt^d` array containing the `T::Integer` weights of the symmetrized PTR
quadrature, aka the Monkhorst Pack weights, `flag` a tuple of boolean arrays that indicate
whether a slice of `wsym` has nonzero entries, and `nsym` the number of non-zero weights.
(see Algorithm 3. in [Kaye et al.](http://arxiv.org/abs/2211.12959)).
The algorithm is serial and has time and memory requirements scaling as O(npt^d).

The arithmetic precision for the calculation is inferred from the element type of `syms` and
the integer size needs to be increased proportionally to the size of `syms`. The default
value of `UInt8` is sufficient for the 48 possible symmetries in 3d.
"""
function symptr_rule(npt::Int, ndim::Val{d}, syms, ::Type{T}=UInt8) where {d,T<:Integer}
    length(syms)-1 <= typemax(T) || throw(ArgumentError("$(length(syms)) symmetries overflow $T weights"))
    ((F = eltype(eltype(syms))) <: Real) || throw(ArgumentError("non-real arithmetic precision determined from symmetry operators"))
    x = F(2)*ptrpoints(npt) .- F(1)
    wsym = ones(T, ntuple(_ -> npt, ndim))
    flags = make_flags(npt, ndim)
    nsym = symptr_rule!!(wsym, flags, x, syms, npt)
    return wsym, flags, nsym
end

make_flags(::Int, ::Val{0}) = ()
function make_flags(npt::Int, ::Val{d}) where {d}
    ndim = Val(d-1)
    T = NTuple{3,Int}
    A = Array{T,d-1}(undef, ntuple(_ -> npt, ndim))
    return (fill!(A, (0,0,0)), make_flags(npt, ndim)...)
end

function symptr_rule!!(wsym, flags, x, syms, npt)
    flag, f = flags[begin:end-1], flags[end]
    nsym, j, k = _symptr_rule!!(wsym, flag, x, syms, npt, 0, ())
    if nsym != 0
        f[] = (1, j, k)
    end
    return nsym
end

# CONJECTURE: this algorithm produces locally convex ibz points, i.e. z is convex, and for
# each z slice y is convex, and so on
function _symptr_rule!!(wsym::Array{T,d}, ::Tuple{}, x, syms, npt, nsym, idx) where {T,d}
    F = eltype(eltype(syms))
    m = n = 0
    for i_1 in 1:npt
        i = CartesianIndex((i_1, idx...))
        if (@inbounds isone(wsym[i]))
            m == 0 && (m = i_1) # entering IBZ (convex)
        else
            m == 0 && continue  # waiting for IBZ
            n == 0 && (n = i_1-1)
            continue
            # break    # exiting IBZ (convex)
        end
        nsym += 1
        for S in syms
            x_i = SVector{d,eltype(x)}(ntuple(k -> x[i.I[k]], Val(d)))
            xsym = S * x_i
            iii = CartesianIndex(ntuple(Val(d)) do k
                # Here we fold the symmetrized kpt back to the FBZ
                if F <: AbstractFloat
                    u_k = F(0.5) * (xsym[k] + F(1))
                    ii_k = npt * (u_k - floor(u_k)) + F(1)
                    iii_k = round(Int, ii_k)
                    abs(iii_k - ii_k) > sqrt(eps(F)) && throw("Inexact index")
                    return iii_k
                else    # exact arithmetic
                    return Int((npt * mod(xsym[k] + 1, 2) + 2) // 2)
                end
            end)
            any(>(npt), iii.I) && continue
            i == iii && continue
            @inbounds if isone(wsym[iii])
                wsym[iii] = zero(T)
                wsym[i] += one(T)
            end
        end
    end
    return nsym, m, n
end

function _symptr_rule!!(wsym::Array, flags::Tuple, x, syms, npt, nsym, idx)
    flag, f = flags[begin:end-1], flags[end]
    m = n = 0
    for i in 1:npt
        ntmp = nsym
        ii = (i, idx...)
        nsym, j, k = _symptr_rule!!(wsym, flag, x, syms, npt, nsym, ii)
        if ntmp != nsym # interior of IBZ (convex)
            m == 0 && (m = i)
            @inbounds f[CartesianIndex(ii)] = (ntmp + 1, j, k)
        elseif m == 0   # waiting to enter the IBZ
            continue
        else            # exiting the IBZ
            n == 0 && (n = i-1)
            continue
            # break # should be safe by convexity
        end
    end
    return nsym, m, n
end

# rule interface
function MonkhorstPack(::Type{T}, ndim::Val{d}, npt, syms) where {d,T}
    u = ptrpoints(T, npt)
    wsym, flag, nsym = symptr_rule(npt, ndim, syms)
    wx = Vector{Tuple{eltype(wsym),SVector{d,eltype(u)}}}(undef, nsym)
    n = 0
    for i in CartesianIndices(wsym)
        n < nsym || break
        @inbounds w = wsym[i]
        iszero(w) && continue
        @inbounds wx[n += 1] = (w, ntuple(j -> u[i[j]], ndim))
    end
    return MonkhorstPack(npt, length(syms), wx)
end

function (r::MonkhorstPackRule)(::Type{T}, v::Val{d}) where {T,d}
    return MonkhorstPack(T, v, r.n₀, r.syms)
end

function nextrule(p::MonkhorstPack{d,T}, r::MonkhorstPackRule) where {d,T}
    return MonkhorstPack(T, Val(d), p.npt+r.Δn, r.syms)
end

function (rule::MonkhorstPack{d})(f::F, B::Basis, buffer=nothing) where {d,F}
    arule = AffineQuad(rule, B)
    return quadsum(arule, f, arule.vol / (rule.npt^d * rule.nsyms), buffer)
end
