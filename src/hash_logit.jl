type HashLogitOptions <: ClassificationModelOptions
    bits::Int
    alpha::Float64
end

function hash_logit_options(;bits::Int=18,
                             alpha::Float64=0.1)
    HashLogitOptions(bits, alpha)
end

type HashLogit <: ClassificationModel
    weights::Vector{Float64}
    counts::Vector{Float64}
    options::HashLogitOptions
end

function StatsBase.fit(xs, ys, opts::HashLogitOptions)
    hash_logit = HashLogit(zeros(2^opts.bits), zeros(2^opts.bits), opts)
    for i=1:length(xs)
        fit!(hash_logit, xs[i], ys[i])
    end
end

hash_features(x, bits::Int) = [(hash(f) >> (64 - bits)) for f=x]

function fit!(hash_logit::HashLogit, x, y)
    features = hash_features(x, hash_logit.opts.bits)
end

function StatsBase.predict(hash_logit::HashLogit, features::Vector{Int})
    pred = 0.0
    for f=feas
        pred += model.weights[f+1]
    end
    1 / (1. + exp(-max(min(pred, 20.), -20.)))
end

function learn!(model::LogisticModel, d::DataHash)
    pred = predict(model, d.features)
    update = (pred-d.label)*model.alpha
    for f = d.features
        model.weights[f+1] -= update / sqrt(model.counts[f+1]+1)
        model.counts[f+1] += 1.0
    end
    pred
end

function learn!(model::LogisticModel, d::DataPoint)
    learn!(model, hash_features(model, d))
end

function learn!(model::LogisticModel, df::DataFrame, features)
    loss = zeros(nrow(df))
    for i=1:nrow(df)
        pred = learn!(model, DataPoint(df[:Label][i], [join([string(f),string(df[f][i])]) for f=features]))
        loss[i] = log_loss(df[:Label][i], pred)
    end
    println("Mean Loss: ", mean(loss))
end
