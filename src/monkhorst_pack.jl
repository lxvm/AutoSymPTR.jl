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
"""
    symptr_rule(npt, ::Val{d}, syms, T::Integer=UInt8) where {d}

Returns `wsym`, a `npt^d` array containing the `T::Integer` weights of the symmetrized PTR
quadrature, aka the Monkhorst Pack weights, and `nsym` the number of non-zero weights.
(see Algorithm 3. in [Kaye et al.](http://arxiv.org/abs/2211.12959)).

The arithmetic precision for the calculation is inferred from the element type of `syms` and
the integer size needs to be increased proportionally to the size of `syms`. The default
value of `UInt8` is sufficient for the 48 possible symmetries in 3d.
"""
function symptr_rule(npt::Int, ::Val{N}, syms, ::Type{T}=UInt8) where {N,T<:Integer}
    length(syms)-1 <= typemax(T) || throw(ArgumentError("$(length(syms)) symmetries overflow $T weights"))
    ((F = eltype(eltype(syms))) <: Real) || throw(ArgumentError("non-real arithmetic precision determined from symmetry operators"))
    x = F(2)*ptrpoints(npt) .- F(1)
    nsym = 0
    wsym = ones(T, ntuple(_ -> npt, Val(N)))
    for i in CartesianIndices(wsym)
        isone(wsym[i]) || continue
        nsym += 1
        for S in syms
            x_i = SVector{N,eltype(x)}(ntuple(k -> x[i.I[k]], Val(N)))
            xsym = S * x_i
            iii = CartesianIndex(ntuple(Val(N)) do k
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
            if isone(wsym[iii])
                wsym[iii] = zero(T)
                wsym[i] += one(T)
            end
        end
    end
    return wsym, nsym
end

# rule interface
function MonkhorstPack(::Type{T}, ndim::Val{d}, npt, syms) where {d,T}
    u = ptrpoints(T, npt)
    wsym, nsym = symptr_rule(npt, ndim, syms)
    wx = Vector{Tuple{eltype(wsym),SVector{d,eltype(u)}}}(undef, nsym)
    n = 0
    for i in CartesianIndices(wsym)
        n < nsym || break
        iszero(wsym[i]) && continue
        wx[n += 1] = (wsym[i], ntuple(j -> u[i[j]], ndim))
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
