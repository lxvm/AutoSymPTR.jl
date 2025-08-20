
struct BatchIntegrand{F,Y,X}
    # in-place function f!(y, x) that takes an array of x values and outputs an array of results in-place
    f!::F
    y::Y
    x::X
    max_batch::Int # maximum number of x to supply in parallel (defaults to typemax(Int))
    function BatchIntegrand(f!, y::AbstractArray, x::AbstractVector, max_batch::Integer=typemax(Int))
        max_batch > 0 || throw(ArgumentError("maximum batch size must be positive"))
        return new{typeof(f!),typeof(y),typeof(x)}(f!, y, x, max_batch)
    end
end

BatchIntegrand(f!, y, x; max_batch::Integer=typemax(Int)) =
    BatchIntegrand(f!, y, x, max_batch)

function _evalrule(rule, f::BatchIntegrand)
    # unroll first batch iteration to get right types
    n = length(rule)
    m = min(n, f.max_batch)
    resize!(f.x, m); resize!(f.y, m)
    prev = next = iterate(rule)
    next === nothing && throw(ArgumentError("empty rule"))
    i = j = 0
    while next !== nothing && i < m
        (_,x), state = next
        f.x[i += 1] = x
        next = iterate(rule, state)
    end
    f.f!(f.y, f.x)
    (w,_), state = prev
    I = mymul(w, f.y[j += 1])
    prev = iterate(rule, state)
    while prev !== nothing && j < m
        (w,_), state = prev
        I += mymul(w, f.y[j += 1])
        prev = iterate(rule, state)
    end
    # accumulate remainder
    while j < n
        if i == j
            while next !== nothing && i-j < m
                (_,x), state = next
                f.x[(i += 1) - j] = x
                next = iterate(rule, state)
            end
            if next === nothing
                resize!(f.x, i-j)
                resize!(f.y, i-j)
            end
            f.f!(f.y, f.x)
        else
            k = j
            while prev !== nothing && j < i
                (w,_), state = prev
                I += mymul(w, f.y[(j += 1) - k])
                prev = iterate(rule, state)
            end
        end
    end

    return I
end
