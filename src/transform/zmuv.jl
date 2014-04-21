# Zero mean, unit variance transformer
type ZmuvOptions <: TransformerOptions
end

type Zmuv <: Transformer
    means::Matrix{Float64}
    stds::Matrix{Float64}
end

function StatsBase.fit(x::Matrix{Float64}, opts::ZmuvOptions)
    means = mean(x, 1)
    stds = std(x, 1)
    stds[isnan(stds)] = 0.0
    Zmuv(means, stds)
end

function transform(zmuv::Zmuv, x::Vector{Float64})
    vec(transform(zmuv, x'))
end

function transform(zmuv::Zmuv, x::Matrix{Float64})
    res = broadcast(/, broadcast(-, x, zmuv.means), zmuv.stds)
    res[isnan(res)] = 0.0
    res[isinf(res)] = 0.0
    res
end