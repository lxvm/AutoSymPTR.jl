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


export symptr, autosymptr, ptr, autoptr

"""
    npt_update(f, npt, [increment=50])

Returns `npt + increment` to try and get another digit of accuracy from PTR.

!!! note "For developers"
    This fallback option is a heuristic, since the scaling of the error is
    generally problem-dependent, so it is appropriate to specialize this method
    based on the integrand type.
"""
npt_update(f, npt::Integer, increment::Integer=50) = npt + increment


include("ptr.jl")
include("symptr.jl")


# this is the PTR convergence loop
function do_autosymptr(f, B, ::Val{d}, (; npt1, rule1, npt2, rule2), atol, rtol, maxevals, nrm, syms) where d
    nsyms = (syms===nothing) ? 1 : length(syms)

    if (numevals = length(rule1) + length(rule2)) == 0
        # initialize rules
        npt1[] = npt_update(f, 0)
        npt2[] = npt_update(f, npt1[])
        (npt1[]^d + npt2[]^d)/nsyms ≥ maxevals && throw(ArgumentError("initial npts exceeds maxevals=$maxevals"))
        rule1=symptr_rule!(rule1, npt1[], Val(d), syms)
        rule2=symptr_rule!(rule2, npt2[], Val(d), syms)
    else
        numevals ≥ maxevals && throw(ArgumentError("initial npts exceeds maxevals=$maxevals"))
    end

    int1 = symptr(f, B, syms, npt = npt1[], rule = rule1)
    int2 = symptr(f, B, syms, npt = npt2[], rule = rule2)
    err = nrm(int1 - int2)

    while true
        (err ≤ max(rtol*nrm(int2), atol) || !isfinite(err)) && break
        if numevals ≥ maxevals
            @warn "maxevals exceeded during convergence"
            return int2, err
        end
        # update coarse result with finer result
        int1 = int2
        npt1[] = npt2[]
        copy!(rule1, rule2)
        # evaluate integral on finer grid
        npt2[] = npt_update(f, npt1[])
        symptr_rule!(rule2, npt2[], Val(d), syms)
        int2 = symptr(f, B, syms, npt = npt2[], rule = rule2)
        numevals += length(rule2.x)
        # self-convergence error estimate
        err = nrm(int1 - int2)
    end
    return int2, err
end


"""
    autosymptr(f, B::AbstractMatrix, syms; atol=0, rtol=sqrt(eps()), maxevals=typemax(Int64), buffer=nothing)

Computes the integral of `f` to within the specified tolerances and returns a
tuple `(I, E, numevals, rules)` containing the estimated integral, the estimated
error, the number of integrand evaluations. The integral on the symmetrized
domain needs to be mapped back to the full domain by the caller, since for
array-valued integrands this depends on the representation of the integrand
under the action of the symmetries.

!!! note "Convergence depends on periodicity"
    If the routine takes a long time to return, double check at the period of
    the function `f` along the basis vectors in the columns of `B` is consistent.
"""
function autosymptr(f, B::AbstractMatrix, syms, rule=symptr_rule; atol=nothing, rtol=nothing, maxevals=typemax(Int64), norm=norm, buffer=nothing)
    d = checksquare(B); T = float(eltype(B))
    nsyms = (syms===nothing) ? 1 : length(syms)
    atol_ = (atol===nothing) ? zero(T) : atol/nsyms # rescale tolerance to correct for symmetrization
    rtol_ = (rtol===nothing) ? (iszero(atol_) ? sqrt(eps(T)) : zero(T)) : rtol
    buffer_ = (buffer===nothing) ? alloc_autobuffer(T, Val(d), rule) : buffer
    do_autosymptr(f, B, Val(d), buffer_, atol_, rtol_, maxevals, norm, syms)
end

"""
    alloc_autobuffer(::Type{T}, ::Val{d}, rule)

Initialize an empty buffer of PTR rules evaluated from `rule(T,Val(d))` where
`T` is the domain type and `d` is the number of dimensions.

!!! note "For developers"
    Providing a special `rule`, (e.g. [`PTRGrid`](@ref))
"""
alloc_autobuffer(::Type{T}, ::Val{d}, rule) where {T,d} =
    (npt1=fill(0), rule1=rule(T,Val(d)), npt2=fill(0), rule2=rule(T,Val(d)))

# aliases for syms===nothing that ignore symmetry altogether

symptr_rule!(rule::PTRRule, npt, ::Val{d}, ::Nothing) where d =
    ptr_rule!(rule, npt, Val(d))

symptr(f, B::AbstractMatrix, ::Nothing; kwargs...) = ptr(f, B; kwargs...)

"""
    autoptr(f, B; kwargs...)

Same as [`autosymptr`](@ref) with trivial symmetries.
"""
autoptr(f, B; kwargs...) = autosymptr(f, B, nothing; kwargs...)
autosymptr(f, B::AbstractMatrix, ::Nothing; kwargs...) =
    autosymptr(f, B, nothing, ptr_rule; kwargs...)

autoptr_buffer(::Type{T}, ::Val{d}) where {T,d} =
    autosymptr_buffer(T, Val(d), ptr_rule)

end