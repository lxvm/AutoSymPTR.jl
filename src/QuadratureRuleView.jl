
struct QuadratureRuleView{Q <: AbstractQuadratureRule, I<:UnitRange} <: AbstractQuadratureRule
    rule::Q
    indices::I
    function QuadratureRuleView(rule::Q, indices::I) where {Q <: AbstractQuadratureRule, I <: UnitRange}
        @assert first(indices) >= firstindex(rule)
        @assert last(indices) <= lastindex(rule)
        return new{Q, I}(rule, indices)
    end
end

Base.view(rule::AbstractQuadratureRule, indices) = QuadratureRuleView(rule, indices)
Base.view(rule::QuadratureRuleView, indices) = QuadratureRuleView(rule.rule, rule.indices[indices])

Base.firstindex(rule::QuadratureRuleView) = firstindex(rule.indices)
Base.lastindex(rule::QuadratureRuleView) = lastindex(rule.indices)
Base.length(rule::QuadratureRuleView) = length(rule.indices)

Base.eltype(::Type{<:QuadratureRuleView{Q}}) where {Q} = eltype(Q)
Base.IndexStyle(::Type{<:QuadratureRuleView{Q}}) where {Q} = Base.IndexStyle(TR)
Base.@propagate_inbounds function Base.getindex(rule::QuadratureRuleView, i::Int)
    ixs = Base.IndexStyle(typeof(rule))
    if ixs isa IndexLinear
        @inbounds rule.rule[rule.indices[i]]
    elseif ixs isa IndexCartesian
        @inbounds rule.rule[ptrindex(size(rule.rule), rule.indices[i])]
    else
        error("unknown index style")
    end
end

function Base.iterate(qv::QuadratureRuleView)
    ixs = Base.IndexStyle(typeof(qv.rule))
    isempty(qv.indices) && return nothing
    i, f = if ixs isa IndexLinear
        first(qv.indices), last(qv.indices)
    elseif ixs isa IndexCartesian
        o = CartesianIndex(ntuple(n -> firstindex(qv.rule, n), Val{ndims(qv.rule)}()))
        o = o - one(o)
        _lin2cart(size(qv.rule), first(qv.indices)) + o, _lin2cart(size(qv.rule), last(qv.indices)) + o
    else
        error("unknown index style")
    end
    item = @inbounds qv.rule[i]
    return item, (i, f)
end
function Base.iterate(qv::QuadratureRuleView, (state, f))
    state == f && return nothing
    ind = nextind(qv.rule, state)
    item = @inbounds qv.rule[ind]
    return item, (ind, f)
end

function _lin2cart(sz::NTuple{N,Int}, i::Int) where {N}
    pow = cumprod(ntuple(n->n==1 ? 1 : sz[n-1], Val(N)))
    pow[1] <= i <= prod(sz) || throw(BoundsError(sz, i))
    idx = ntuple(n -> rem(div(i-1, pow[n]), sz[n])+1, Val(N))
    return CartesianIndex(idx)
end