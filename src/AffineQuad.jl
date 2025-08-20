
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

apply_affine(aq::AffineQuad, x) =  myadd(mymul(aq.A, myadd(x, aq.c)), aq.b)

function Base.iterate(aq::AffineQuad, args...)
    next = iterate(aq.rule, args...)
    next === nothing && return nothing
    (w, x), state = next
    return (w, apply_affine(aq, x)), state
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

evalrule(rule::AffineQuad, f) = _evalrule(rule, f) * rule.vol
