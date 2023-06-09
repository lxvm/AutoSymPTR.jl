# quadrature rule for npt^d PTR reduced by a collection of symmetries
# computes the value of the integral over the IBZ

# nodes and weights
struct MonkhorstPack{d,T}
    npt::Int64
    nsyms::Int64
    wx::Vector{Tuple{Int64,SVector{d,T}}}
end

# indexing interface
Base.getindex(rule::MonkhorstPack, i::Int) = rule.wx[i]

# iteration interface
Base.iterate(rule::MonkhorstPack, args...) = iterate(rule.wx, args...)
Base.length(rule::MonkhorstPack) = length(rule.wx)

"""
    symptr_rule(::Val{d}, npt, syms) where d

Returns `flag`, `wsym`, and `nsym` containing a mask for the nodes of an
`npt` symmetrized PTR quadrature ptr, and the corresponding integer weights
(see Algorithm 3. in [Kaye et al.](http://arxiv.org/abs/2211.12959)).
"""
@generated function symptr_rule(npt::Int, ::Val{N}, syms) where N
    quote
    x = range(-1, 1, length=npt+1)[1:npt]
    nsym = 0
    wsym = Vector{Int}(undef, npt^$N)
    flag = ones(Bool, Base.Cartesian.@ntuple $N _ -> npt)
    Base.Cartesian.@nloops $N i flag begin
        (Base.Cartesian.@nref $N flag i) || continue
        nsym += 1
        wsym[nsym] = 1
        for S in syms
            x_i = Base.Cartesian.@ncall $N SVector{$N,Float64} k -> x[i_k]
            xsym = S * x_i
            Base.Cartesian.@nexprs $N k -> begin
                ii_k = 0.5npt * mod(xsym[k] + 1.0, 2.0) + 1.0
                iii_k = round(Int, ii_k)
                (iii_k - ii_k) > 1e-12 && throw("Inexact index")
            end
            (Base.Cartesian.@nany $N k -> (iii_k > npt)) && continue
            (Base.Cartesian.@nall $N k -> (iii_k == i_k)) && continue
            if (Base.Cartesian.@nref $N flag iii)
                (Base.Cartesian.@nref $N flag iii) = false
                wsym[nsym] += 1
            end
        end
    end
    return flag, wsym, nsym
    end
end

# rule interface
function MonkhorstPack(::Type{T}, v::Val{d}, npt, syms) where {d,T}
    u = ptrpoints(npt)
    flag, wsym, nsym = symptr_rule(npt, Val(d), syms)
    wx = Vector{Tuple{Int64,SVector{d,T}}}(undef, nsym)
    n = 0
    for i in CartesianIndices(flag)
        flag[i] || continue
        n += 1
        wx[n] = (wsym[n], ntuple(n -> u[i[n]], Val(d)))
        n >= nsym && break
    end
    return MonkhorstPack(npt, length(syms), wx)
end
countevals(rule::MonkhorstPack) = length(rule)

function (r::MonkhorstPackRule)(::Type{T}, v::Val{d}) where {T,d}
    return MonkhorstPack(T, v, r.n₀, r.syms)
end

function nextrule(p::MonkhorstPack{d,T}, r::MonkhorstPackRule) where {d,T}
    return MonkhorstPack(T, Val(d), p.npt+r.Δn, r.syms)
end

function (rule::MonkhorstPack{d})(f, B::Basis, buffer=nothing) where d
    return quadsum(rule, f, B, buffer) * ((abs(det(B.B)) / (rule.npt^d * rule.nsyms)))
end
