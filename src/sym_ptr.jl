"""
    npt_update(f, npt, [increment=50])

Returns `npt + increment` to try and get another digit of accuracy from PTR.
This fallback option is a heuristic, since the scaling of the error is generally
problem-dependent, so it is appropriate to specialize this method based on the
integrand type 
"""
npt_update(f, npt::Integer, increment::Integer=50) = npt + increment

"""
    ptr(npt, f, ::Val{d}, ::Type{T}, syms) where {d,T}

Returns the grid points and weights, `x,w`, to use for an `npt` PTR quadrature
of `f` on a domain of dimension `d` and type `T` while applying the relevant
symmetries `syms` to reduce the number of evaluation points.

!!! note "For developers"
    Dispatching on the type of `f` allows for customized ptr rules, which must
    implement [`ptr`](@ref), [`ptr!`](@ref), and [`evalptr`](@ref).
"""
ptr(npt, _, ::Val{d}, ::Type{T}, syms) where {d,T} =
    ptr(npt, Val(d), T, syms)


"""
    ptr(npt, ::Val{d}, ::Type{T}, syms) where {d,T}

Returns the grid points and weights, `x,w`, to use for an `npt` PTR quadrature
of `f` on the unit cube `d` and type `T` while applying the relevant
symmetries `syms` to reduce the number of evaluation points.
"""
function ptr(npt, ::Val{d}, ::Type{T}, syms) where {d,T}
    x = Vector{SVector{d,T}}(undef, 0)
    w = Vector{Int}(undef, 0)
    rule = (; x=x, w=w)
    ptr!(rule, npt, Val(d), T, syms)
end

"""
    ptr!(rule, npt, f, ::Val{d}, ::Type{T}, syms) where {d,T}

In-place version of [`ptr`](@ref).
"""
ptr!(rule, npt, _, ::Val{d},::Type{T}, syms) where {T,d} =
    ptr!(rule, npt, Val(d), T, syms)

"""
    ptr!(rule, npt, ::Val{d}, ::Type{T}, syms) where {d,T}
"""
function ptr!(rule, npt, ::Val{d}, ::Type{T}, syms) where {d,T}
    flag, wsym, nsym = ptr_(Val(d), npt, syms)
    u = range(0, 1, length=npt+1)[1:npt]
    resize!(rule.x, nsym)
    resize!(rule.w, nsym)
    n = 0
    for i in CartesianIndices(flag)
        flag[i] || continue
        n += 1
        rule.x[n] = convert(SVector{d,T}, ntuple(n -> u[i[n]], Val(d)))
        rule.w[n] = wsym[n]
        n >= nsym && break
    end
    rule
end


"""
    ptr_(::Val{d}, npt, syms) where d

Returns `flag`, `wsym`, and `nsym` containing a mask for the nodes of an
`npt` symmetrized PTR quadrature ptr, and the corresponding integer weights
(see Algorithm 3. in [Kaye et al.](http://arxiv.org/abs/2211.12959)).

!!! note
    This routine computes the IBZ  For getting the full integral
    from this result, use [`symmetrize`](@ref)
"""
@generated function ptr_(::Val{N}, npt::Int, syms) where N
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
    evalptr(rule, npt, f, B::SMatrix{d,d}, syms) where d

Evaluates the `npt^d` point PTR `rule` on the integrand `f`. The coordinates are
mapped into the basis `B`, whose basis vectors are stored as columns of the
matrix. The integral is returned, and if the symmetries are nontrivial, the
integral on the symmetry-reduced domain is returned, which the caller needs to
remap to the full domain
"""
function evalptr(rule, npt, f, B::SMatrix{d,d}, syms) where d
    # X = Base.promote_op(*, typeof(B), eltype(rule.x))
    # T = Base.promote_op(ptr_integrand, typeof(f), X)
    # int = sum(i -> rule.w[i]*ptr_integrand(f, B*rule.x[i]), 1:length(rule.x); init=zero(T))
    int = mapreduce((w, x) -> w*ptr_integrand(f, B*x), +, rule.w, rule.x)
    int * det(B)/npt^d/length(syms)
end

"""
    ptr_integrand(f, x) = f(x)

!!! note "For developers"
    The caller may dispatch on the type of `f` if they would like to specialize
    this function together with [`ptr`](@ref) so that `x` is
    a more useful precomputation (e.g. a Fourier series evaluated at `x`).
"""
ptr_integrand(f, x) = f(x)

"""
    ptrcopy!(rule1, rule2)

Copies the values of in the `NamedTuple` `rule2` to `rule1`, which must all be
mutable.
"""
function ptrcopy!(rule1, rule2)
    for key in keys(rule1)
        copy!(rule1[key], rule2[key])
    end
    rule1
end

"""
    symptr(f, B::AbstractMatrix, syms=nothing; npt=npt_update(f,0), rule=ptr(npt,f,B,syms))

Computes an `npt` symmetrized PTR rule of `f` on a unit hypercube in basis `B`
and returns a tuple `(I, rules)` the integral, `I`, and the PTR `rules` used.
Optionally uses a precomputed keyword `rule`.
"""
function symptr(f, B_::AbstractMatrix{T}, syms=nothing; kwargs...) where T
    d = checksquare(B_)
    B = convert(SMatrix{d,d,float(T),d^2}, B_)
    symptr_(f, B, syms, symptr_kwargs(f, B, syms; kwargs...)...)
end

symptr_kwargs(f, ::SMatrix{d,d,T}, syms; npt=npt_update(f, 0), rule=nothing) where {d,T} =
    (npt=npt, rule=something(rule, ptr(npt, f, Val(d), T, syms)))

symptr_(f, B, syms, npt, rule) =
    (evalptr(rule, npt, f, B, syms), (npt=npt, rule=rule))

"""
    autosymptr(f, B::AbstractMatrix, syms=nothing;
    atol=0, rtol=sqrt(eps()), maxevals=typemax(Int64),
    npt1=npt_update(f,0), npt2=npt_update(f,npt1),
    rule1=ptr(npt1, f, B, syms), rule2=ptr(npt2, f, B, syms))

Computes the integral of `f` to within the specified tolerances and returns a
tuple `(I, E, numevals, rules)` containing the estimated integral, the estimated
error, the number of integrand evaluations, and a `NamedTuple` of `rules` used
to compute the PTR on the most refined grid.

!!! note "Convergence depends on periodicity"
    If the routine takes a long time to return, double check at the period of
    the function `f` along the basis vectors in the columns of `B` is consistent.
"""
function autosymptr(f, B_::AbstractMatrix{T}, syms=nothing; kwargs...) where T
    d = checksquare(B_)
    B = convert(SMatrix{d,d,float(T),d^2}, B_)
    autosymptr_(f, B, syms, autosymptr_kwargs(f, B, syms; kwargs...)...)
end

function autosymptr_kwargs(f, ::SMatrix{d,d,T}, syms;
    atol=nothing, rtol=nothing, maxevals=typemax(Int64),
    npt1=nothing, rule1=nothing, npt2=nothing, rule2=nothing,
) where {d,T}
    npt1 = something(npt1, fill(npt_update(f, 0)))
    npt2 = something(npt2, fill(npt_update(f, only(npt1))))
    nsyms = isnothing(syms) ? 1 : length(syms)
    (npt1[]^d + npt2[]^d)/nsyms ≥ maxevals && throw(ArgumentError("initial npts exceeds maxevals=$maxevals"))
    atol_ = something(atol, zero(T))/nsyms # rescale tolerance to correct for symmetrization
    rtol_ = something(rtol, iszero(atol_) ? sqrt(eps(T)) : zero(T))
    (npt1=npt1, rule1=something(rule1, ptr(npt1[], f, Val(d), T, syms)),
     npt2=npt2, rule2=something(rule2, ptr(npt2[], f, Val(d), T, syms)),
     atol=atol_, rtol=rtol_, maxevals=maxevals)
end

function autosymptr_(f, B::SMatrix{d,d,T}, syms, npt1, rule1, npt2, rule2, atol, rtol, maxevals) where {d,T}
    numevals = length(rule1.x) + length(rule2.x)
    numevals ≥ maxevals && throw(ArgumentError("initial npts exceeds maxevals=$maxevals"))
    int1 = evalptr(rule1, npt1[], f, B, syms)
    int2 = evalptr(rule2, npt2[], f, B, syms)
    err = norm(int1 - int2)
    while true
        (err ≤ max(rtol*norm(int2), atol) || numevals ≥ maxevals || !isfinite(err)) && break
        # update coarse result with finer result
        int1 = int2
        npt1[] = npt2[]
        ptrcopy!(rule1, rule2)
        # evaluate integral on finer grid
        npt2[] = npt_update(f, npt1[])
        ptr!(rule2, npt2[], f, Val(d), T, syms)
        int2 = evalptr(rule2, npt2[], f, B, syms)
        numevals += length(rule2.x)
        # self-convergence error estimate
        err = norm(int1 - int2)
    end
    return int2, err, numevals, (npt1=npt1, rule1=rule1, npt2=npt2, rule2=rule2)
end


# fancy type for PTR grid iteration

"""
    PTRGrid{d}(x::Vector{T}) where {d,T}

Stores a `d` dimensional Cartesian product grid of `SVector{d,T}`.
Similar to `Iterators.product(ntuple(n->x, d)...)`.
Uses the same number of grid points per dimension.
"""
struct PTRGrid{N,T}
    x::Vector{T}
    function PTRGrid{N}(x::Vector{T}) where {N,T}
        @assert N isa Integer
        @assert N >= 1
        new{N,T}(x)
    end
end

Base.ndims(::PTRGrid{N,T}) where {N,T} = N
Base.eltype(::Type{PTRGrid{N,T}}) where {N,T} = SVector{N,T}
Base.length(p::PTRGrid) = length(p.x)^ndims(p)
Base.size(p::PTRGrid) = (l=length(p.x); ntuple(n->l, Val(ndims(p))))
Base.copy!(p::PTRGrid, v::AbstractVector) = copy!(p.x, v)
Base.copy!(p::PTRGrid, q::PTRGrid) = copy!(p, q.x)

# map a linear index to a Cartesian index
function ptrindex(p::PTRGrid, i::Int)
    npt = length(p.x)
    pow = cumprod(ntuple(n->n==1 ? 1 : npt, Val(ndims(p))))
    pow[1] <= i <= npt*pow[ndims(p)] || throw(BoundsError(p, i))
    CartesianIndex(ntuple(n -> rem(div(i-1, pow[n]), npt)+1, Val(ndims(p))))
end

# map a Cartesian index to a linear index
function ptrindex(p::PTRGrid{N}, i::CartesianIndex{N}) where N
    npt = length(p.x)
    idx = 0
    for j in reverse(i.I)
        1 <= j <= npt || throw(BoundsError(p, i.I))
        idx *= npt
        idx += j-1
    end
    idx+1
end

function Base.getindex(p::PTRGrid{N,T}, i::Int) where {N,T}
    idx = ptrindex(p, i)
    SVector{N,T}(ntuple(n -> p.x[idx[n]], Val(N)))
end
Base.getindex(p::PTRGrid{N,T}, idx::CartesianIndex{N}) where {N,T} =
    SVector{N,T}(ntuple(n -> p.x[idx[n]], Val(N)))

Base.isdone(p::PTRGrid{N,T}, state) where {N,T} = !(1 <= state <= length(p.x)^N)
function Base.iterate(p::PTRGrid{N,T}, state=1) where {N,T}
    Base.isdone(p, state) && return nothing
    (p[state], state+1)
end


# special cases for syms=nothing with trivial weights for npt^d PTR

function ptr(npt, ::Val{d}, ::Type{T}, syms::Nothing) where {d,T}
    x = Vector{T}(undef, 0)
    rule = (; x=PTRGrid{d}(x))
    ptr!(rule, npt, Val(d), T, syms)
end

function ptr!(rule, npt, ::Val{d}, ::Type{T}, ::Nothing) where {d,T}
    copy!(rule.x, range(0, 1, length=npt+1)[1:npt])
    rule
end

function evalptr(rule, npt, f, B::SMatrix{d,d}, ::Nothing) where d
    int = sum(x -> ptr_integrand(f, B*x), rule.x)
    int * det(B)/npt^d
end