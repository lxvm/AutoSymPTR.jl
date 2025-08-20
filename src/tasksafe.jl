
"""
    TaskSafeIntegrand(f, [channel=nothing]; kws...)

Construct an integrand that evaluates quadrature of `f` in parallel on multiple tasks.
`f(x)` is assumed to be thread-safe and free of race conditions, giving the library permission to spawn tasks that simultaneously call `f`.
If evaluation of `f` requires task-local storage, it can be provided in a `channel`, and the api is `f(take!(channel), x)`.
Typically, the channel should have as many elements as threads on the computer.
Lastly, the keyword arguments are the same as those from `ChunkSplitters.chunks` and are used to split the quadrature rule into smaller parts.
Set `n` to the desired number of tasks, typically some multiple of the number of threads to notice an improvement from dynamic load balancing, and `minsize` to increase the cost of the base case.
The `split` keyword must be `Consecutive()` (default) as `RoundRobin()` is currently unsupported.

`TaskSafeIntegrand` is preferrable to `BatchIntegrand` for embarrasingly parrallel quadrature rules due to the fact the summation of the quadrature is performed in parallel and without temporary buffers.
"""
struct TaskSafeIntegrand{F,C<:Union{Nothing,Channel},K}
    f::F
    channel::C
    kws::K
end
TaskSafeIntegrand(f, ch=nothing; kws...) = TaskSafeIntegrand(f, ch, kws)
_evalrule(rule, f::TaskSafeIntegrand) = _taskevalrule(rule, f.f, f.channel; f.kws...)
function _taskevalrule(rule, f::F, maybechannel; kws...) where {F}
    tasks = map(chunks(rule; kws...)) do rchunk
        Threads.@spawn _sumrulechunk(f, rchunk, maybechannel)
    end
    sum(fetch, tasks)
end
function _sumrulechunk(f, rchunk)
    next = Iterators.peel(rchunk)
    next === nothing && error("empty chunk")
    (w0, x0), rest = next
    I = mymul(w0, f(x0))
    for (w, x) in rest
        I += mymul(w, f(x))
    end
    return I
end
_sumrulechunk(f::F, rchunk, ::Nothing) where {F} = _sumrulechunk(f, rchunk)
function _sumrulechunk(f::F, rchunk, channel::Channel) where {F}
    worker = take!(channel)
    try
        _sumrulechunk(x -> f(worker, x), rchunk)
    catch e
        rethrow(e)
    finally
        put!(channel, worker)
    end
end