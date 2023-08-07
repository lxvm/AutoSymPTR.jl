"""
A package implementing the automatic symmetrized periodic trapezoidal rule (PTR)
for spectrally-accurate integration of periodic functions with the algorithm
described by [Kaye et al.](http://arxiv.org/abs/2211.12959). The main routine is
[`autosymptr`](@ref), which computes the integral of a periodic function to
within a requested error tolerance. If nontrivial symmetries are provided, the
routines return the integral on the symmetry-reduced domain, since it is
generally up to the caller to implement the mapping back to the full domain,
which is nontrivial in cases with non-scalar integrands. A simple example is

    using LinearAlgebra, AutoSymPTR
    autosymptr(x -> 1 + mapreduce(y -> cospi(2y), +, x), I(3))[1] ≈ 1
"""
module AutoSymPTR

using LinearAlgebra: norm, det, checksquare
using StaticArrays: SVector, SMatrix
using ChunkSplitters: chunks

export autosymptr, Basis

include("definitions.jl")
include("ptr.jl")
include("monkhorst_pack.jl")


function nextrule!(cache::Vector, state, keepmost, prev, ruledef)
    length(cache) < state || return @inbounds(cache[state]), state+1
    length(cache) == keepmost && (popfirst!(cache); state -= 1)
    next = nextrule(prev, ruledef)
    push!(cache, next)
    return next, state+1
end

# this is the p-adaptive convergence loop
# the cache is a vector of rules of different orders
# the (optional) buffer is a workspace for the rule to store function
# evaluations or accumulators for parallel evaluation
function p_adapt(f::F, dom, ruledef, cache::Vector, keepmost, abstol, reltol, maxevals, nrm, buffer) where {F}
    # unroll first two rule evaluations to get right types
    next = iterate(cache)
    next === nothing && throw(ArgumentError("rule cache is empty"))
    rule_, state = next
    _rule, state = nextrule!(cache, state, keepmost, rule_, ruledef)

    if f isa InplaceIntegrand # rule evaluators should write inplace to f.I
        int_ = f.Itmp .= rule_(f, dom, buffer)
        _int = _rule(f, dom, buffer)
        int_ .-= _int
        err = nrm(int_)
    else
        int_ = rule_(f, dom, buffer)
        _int = _rule(f, dom, buffer)
        err = nrm(int_ - _int)
    end
    numevals = length(rule_) + length(_rule)

    # logic to handle dimensional quantities
    atol = something(abstol, zero(err))
    rtol = something(reltol, iszero(atol) ? sqrt(eps(typeof(one(err)))) : zero(typeof(one(err))))

    while true
        if isnan(err) || isinf(err)
            throw(DomainError(dom, "integrand produced $err in the domain"))
        elseif err ≤ max(rtol*nrm(_int), atol)
            return _int, err
        elseif numevals ≥ maxevals
            @warn "maxevals exceeded"
            return _int, err
        end
        # update coarse result with finer result
        rule_ = _rule
        # evaluate integral on finer grid
        _rule, state = nextrule!(cache, state, keepmost, rule_, ruledef)
        if f isa InplaceIntegrand
            int_ .= _int
            int_ .-= _rule(f, dom, buffer) # _int updated inplace
            err = nrm(int_)
        else
            int_ = _int
            _int = _rule(f, dom, buffer)
            err = nrm(int_ - _int)
        end
        numevals += length(_rule)
    end

    return _int, err
end


function pquadrature(f, dom, ruledef; abstol=nothing, reltol=nothing, maxevals=typemax(Int64), norm=norm, cache=nothing, keepmost::Integer=2, buffer=nothing)
    d = ndims(dom); T = typeof(float(real(one(eltype(dom)))))
    cach = (cache===nothing) ? alloc_cache(T, Val(d), ruledef) : cache
    return p_adapt(f, dom, ruledef, cach, keepmost, abstol, reltol, maxevals, norm, buffer)
end
function pquadrature(f::InplaceIntegrand{F,TI,<:AbstractArray{Nothing}}, dom, ruledef; kws...) where {F,TI}
    d = ndims(dom); T = typeof(float(real(one(eltype(dom)))))
    g = InplaceIntegrand(f.f!, f.I, f.Itmp, f.I/det(one(SMatrix{d,d,T,d^2})*oneunit(eltype(dom))))
    return pquadrature(g, dom, ruledef; kws...)
end

"""
    alloc_cache(::Type{T}, ::Val{d}, rule)

Initialize an empty buffer of PTR rules evaluated from `rule(T,Val(d))` where
`T` is the domain type and `d` is the number of dimensions.

!!! note "For developers"
    Providing a special `rule`, (e.g. [`PTR`](@ref))
"""
function alloc_cache(::Type{T}, ::Val{d}, rule) where {T,d}
    return [rule(T,Val(d))]
end


"""
    autosymptr(f, B::AbstractMatrix, [syms=nothing]; atol=0, rtol=sqrt(eps()), maxevals=typemax(Int64), buffer=nothing)

Computes the integral of `f` to within the specified tolerances and returns a
tuple `(I, E, numevals, rules)` containing the estimated integral, the estimated
error, the number of integrand evaluations. The integral on the symmetrized
domain needs to be mapped back to the full domain by the caller, since for
array-valued integrands this depends on the representation of the integrand
under the action of the symmetries.

Note that if a vector buffer is provided to store integrand evaluations, the integrand
evaluations will be parallelized and it will be assumed that the integrand is threadsafe.

!!! note "Convergence depends on periodicity"
    If the routine takes a long time to return, double check that the period of
    the function `f` along the basis vectors in the columns of `B` is consistent.
"""
function autosymptr(f, dom::Basis; syms=nothing, a=1.0, rule=nothing, abstol=nothing, reltol=nothing, maxevals=typemax(Int64), norm=norm, cache = nothing, keepmost::Integer=2, buffer=nothing)
    rule_ = (rule===nothing) ? MonkhorstPackRule(syms, a) : rule
    return pquadrature(f, dom, rule_, abstol = abstol, reltol = reltol, norm = norm, maxevals = maxevals, cache = cache, keepmost=keepmost, buffer = buffer)
end

end
