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
using StaticArrays: SVector


export autosymptr, Basis

include("definitions.jl")
include("ptr.jl")
include("monkhorst_pack.jl")


function nextrule!(cache::Vector, state, prev, ruledef)
    length(cache) < state || return @inbounds(cache[state]), state+1
    next = nextrule(prev, ruledef)
    push!(cache, next)
    return next, state+1
end

# this is the p-adaptive convergence loop
# the cache is a vector of rules of different orders
# the (optional) buffer is a vector for storing function evaluations
function p_adapt(f, dom, ruledef, cache::Vector, atol, rtol, maxevals, nrm, buffer)
    next = iterate(cache)
    next === nothing && throw(ArgumentError("rule cache is empty"))
    rule_, state = next
    int_ = rule_(f, dom, buffer)
    numevals = countevals(rule_)

    while true
        # evaluate integral on finer grid
        _rule, state = nextrule!(cache, state, rule_, ruledef)
        _int = _rule(f, dom, buffer)
        numevals += countevals(_rule)
        # error estimate
        err = nrm(int_ - _int)
        if isnan(err) || isinf(err)
            throw(DomainError(dom, "integrand produced $err in the domain"))
        elseif err ≤ max(rtol*nrm(_int), atol)
            return _int, err
        elseif numevals ≥ maxevals
            @warn "maxevals exceeded"
            return _int, err
        else
            continue
        end
        # update coarse result with finer result
        int_ = _int
        rule_ = _rule
    end
end


function pquadrature(f, dom, ruledef; abstol=nothing, reltol=nothing, maxevals=typemax(Int64), norm=norm, cache=nothing, buffer=nothing)
    d = ndims(dom); T = typeof(float(real(one(eltype(dom)))))
    atol = (abstol===nothing) ? zero(T) : abstol
    rtol = (reltol===nothing) ? (iszero(atol) ? sqrt(eps(one(T))) : zero(T)) : reltol
    cach = (cache===nothing) ? alloc_cache(T, Val(d), ruledef) : cache
    return p_adapt(f, dom, ruledef, cach, atol, rtol, maxevals, norm, buffer)
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
function autosymptr(f, dom::Basis; syms=nothing, a=1.0, rule=nothing, abstol=nothing, reltol=nothing, maxevals=typemax(Int64), norm=norm, cache = nothing, buffer=nothing)
    T = float(real(eltype(dom)))
    rule_ = (rule===nothing) ? MonkhorstPackRule(syms, a) : rule
    atol = (abstol===nothing) ? zero(T) : abstol/nsyms(rule_) # rescale tolerance to correct for symmetrization
    rtol = (reltol===nothing) ? (iszero(atol) ? sqrt(eps(T)) : zero(T)) : reltol
    return pquadrature(f, dom, rule_, abstol = atol, reltol = rtol, norm = norm, maxevals = maxevals, cache = cache, buffer = buffer)
end

end
