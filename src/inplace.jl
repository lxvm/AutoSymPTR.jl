struct InplaceIntegrand{F,T<:AbstractArray,Y<:AbstractArray}
    # in-place function f!(y, x) that takes one x value and outputs an array of results in-place
    f!::F
    I::T
    Itmp::T
    y::Y
    ytmp::Y
end

"""
    InplaceIntegrand(f!, result::AbstractArray)

Constructor for a `InplaceIntegrand` accepting an integrand of the form `f!(y,x)`. The
caller also provides an output array needed to store the result of the quadrature.
Intermediate `y` arrays are allocated during the calculation, and the final result is
may or may not be written to `result`, so use the IntegralSolution immediately after the
calculation to read the result, and don't expect it to persist if the same integrand is used
for another calculation.
"""
function InplaceIntegrand(f!, result)
    return InplaceIntegrand(f!, result, similar(result), similar(result, Nothing), similar(result, Nothing))
end

function _evalrule(rule, f::InplaceIntegrand)
    next = iterate(rule)
    next === nothing && throw(ArgumentError("empty rule"))
    (w, x), state = next
    f.f!(f.y, x)
    I = f.ytmp .= mymul.(w, f.y)
    next = iterate(rule, state)
    while next !== nothing
        (w, x), state = next
        f.f!(f.y, x)
        I .+= mymul.(w, f.y)
        next = iterate(rule, state)
    end
    return f.I .= I
end

function _evalrule(rule, f::InplaceIntegrand{F,<:AbstractArray,<:AbstractArray{Nothing}}) where {F}
    y = similar(f.I)
    g = InplaceIntegrand(f.f!, f.I, f.Itmp, y, similar(y))
    return _evalrule(rule, g)
end
