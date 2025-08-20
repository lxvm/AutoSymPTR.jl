
# oop-place function f!(take!(channel), x) that takes an array of x values and outputs an array of results in-place
struct ChannelIntegrand{F,C<:Channel,K}
    f::F
    channel::C
    kws::K
end
ChannelIntegrand(f, ch::Channel; kws...) = ChannelIntegrand(f, ch, kws)
function _evalrule(rule, f::ChannelIntegrand)
    channel = f.channel
    _f = f.f
    tasks = map(chunks(rule; f.kws...)) do rchunk
        Threads.@spawn begin
            worker = take!(channel)
            try
                next = Iterators.peel(rchunk)
                next === nothing && error("empty chunk")
                (w0, x0), rest = next
                I = mymul(w0, _f(worker, x0))
                for (w, x) in rest
                    I += mymul(w, _f(worker, x))
                end
                I
            catch e
                rethrow(e)
            finally
                put!(channel, worker)
            end
        end
    end
    sum(fetch, tasks)
end
