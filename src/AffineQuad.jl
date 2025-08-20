
"""
    AffineQuad(rule, A, b, c, vol)

A wrapper to an iterable and indexable quadrature rule that applies an affine coordinate
transformation to the nodes of the form `A*(x+c)+b`. While the weights should be rescaled by the
Jacobian determinant `vol=abs(det(A))`, the value is stored in the `vol` field of the struct
so that the caller can minimize multiplication by applying the rescaling after computing the
rule.

While the quadrature rule should be unitless, the affine transform may have units.
"""
struct AffineQuad{TR <: AbstractQuadratureRule,TA,Tb,Tc,TV} <: AbstractQuadratureRule
    rule::TR
    A::TA
    b::Tb
    c::Tc
    vol::TV
end

_apply_affine(A, b, c, x) =  myadd(mymul(A, myadd(x, c)), b)

function Base.iterate(aq::AffineQuad, args...)
    next = iterate(aq.rule, args...)
    next === nothing && return nothing
    (w, x), state = next
    return (w, _apply_affine(aq.A, aq.b, aq.c, x)), state
end
Base.isdone(aq::AffineQuad, args...) = Base.isdone(aq.rule, args...)
Base.IteratorSize(::Type{<:AffineQuad{TR}}) where {TR} = Base.IteratorSize(TR)
Base.length(aq::AffineQuad) = length(aq.rule)
Base.size(aq::AffineQuad, args...) = size(aq.rule, args...)
Base.IndexStyle(::Type{<:AffineQuad{TR}}) where {TR} = Base.IndexStyle(TR)
# the eltype could change due to the affine transform
Base.eltype(::Type{<:AffineQuad{TR}}) where {TR} = eltype(TR)
Base.view(aq::AffineQuad, indices) = AffineQuad(view(aq.rule, indices), aq.A, aq.b, aq.c, aq.vol)

Base.@propagate_inbounds function Base.getindex(aq::AffineQuad, i)
    w, x = getindex(aq.rule, i)
    return w, apply_affine(aq, x)
end
Base.firstindex(aq::AffineQuad) = firstindex(aq.rule)
Base.lastindex(aq::AffineQuad)  = lastindex(aq.rule)

AffineQuad(rule) = AffineQuad(rule, One(), Zero(), Zero(), 1)
AffineQuad(rule, B::Basis) = AffineQuad(rule, B.B, Zero(), Zero(), abs(det(B.B)))

function evalrule(rule::AffineQuad, f)
    g = _apply_affine2f(rule, f)
    return evalrule(rule.rule, g) * rule.vol
end
function _apply_affine2f(rule::AffineQuad, f::F) where {F}
    A = rule.A
    b = rule.b
    c = rule.c
    x -> f(_apply_affine(A, b, c, x))
end
function _apply_affine2f(rule::AffineQuad, f::InplaceIntegrand)
    A = rule.A
    b = rule.b
    c = rule.c
    f! = f.f!
    g! = (y, x) -> f!(y, _apply_affine(A, b, c, x))
    InplaceIntegrand(g!, f.I, f.Itmp, f.y, f.ytmp)
end
function _apply_affine2f(rule::AffineQuad, f::BatchIntegrand)
    A = rule.A
    b = rule.b
    c = rule.c
    f! = f.f!
    # CAUTION: the following assumes that the affine transformation is unitless
    g! = (y, x) -> f!(y, x .= _apply_affine.(Ref(a), Ref(b), Ref(c), x))
    BatchIntegrand(g!, f.y, f.x; max_batch=f.max_batch)
end
function _apply_affine2f(rule::AffineQuad, f::TaskSafeIntegrand)
    A = rule.A
    b = rule.b
    c = rule.c
    _f = f.f
    g = if f.channel === nothing
        x -> _f(_apply_affine(A, b, c, x))
    else
        (storage, x) -> _f(storage, _apply_affine(A, b, c, x))
    end
    return TaskSafeIntegrand(g, f.channel; f.kws...)
end