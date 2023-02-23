# quadrature rule for npt^d PTR reduced by a collection of symmetries

# nodes and weights
struct SymPTRRule{d,T}
    x::Vector{SVector{d,T}}
    w::Vector{Int}
end
Base.length(r::SymPTRRule) = length(r.x)
function Base.copy!(r::T, s::T) where {T<:SymPTRRule}
    copy!(r.x, s.x)
    copy!(r.w, s.w)
    r
end

function SymPTRRule(::Type{T}, ::Val{d}) where {T,d}
    x = Vector{SVector{d,T}}(undef, 0)
    w = Vector{Int}(undef, 0)
    SymPTRRule(x, w)
end

"""
    symptr_rule!(rule, npt, ::Val{d}, syms) where {d,T}
"""
function symptr_rule!(rule::SymPTRRule, npt, ::Val{d}, syms) where d
    flag, wsym, nsym = symptr_rule(npt, Val(d), syms)
    u = range(0, 1, length=npt+1)[1:npt]
    resize!(rule.x, nsym)
    resize!(rule.w, nsym)
    n = 0
    for i in CartesianIndices(flag)
        flag[i] || continue
        n += 1
        rule.x[n] = ntuple(n -> u[i[n]], Val(d))
        rule.w[n] = wsym[n]
        n >= nsym && break
    end
    rule
end


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
                ii_k = 0.5npt * (xsym[k] + 1.0) + 1.0
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

"""
    symptr(f, B::AbstractMatrix, syms; npt=npt_update(f,0), rule=symptr_rule(float(eltype(B)),npt,checksquare(B),syms))

Evaluates the `npt^d` point PTR `rule` on the integrand `f`. The coordinates are
mapped into the basis `B`, whose basis vectors are stored as columns of the
matrix. The integral is returned, and if the symmetries are nontrivial, the
integral on the symmetry-reduced domain is returned, which the caller needs to
remap to the full domain
"""
function symptr(f, B::AbstractMatrix, syms; npt=npt_update(f, 0), rule=nothing)
    d = checksquare(B); T = float(eltype(B))
    rule_ = (rule===nothing) ? symptr_rule!(SymPTRRule(T, Val(d)), npt, Val(d), syms) : rule
    int = mapreduce((w, x) -> w*f(B*x), +, rule_.w, rule_.x)
    int * det(B)/npt^d/length(syms)
end
