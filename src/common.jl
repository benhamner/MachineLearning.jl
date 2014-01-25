using StatsBase

abstract SupervisedModel
abstract SupervisedModelOptions

abstract ClassificationModel <: SupervisedModel
abstract RegressionModel     <: SupervisedModel

abstract Transformer
abstract TransformerOptions

function predict_probs(model::ClassificationModel, samples::Matrix{Float64})
    probs = Array(Float64, size(samples, 1), length(classes(model)))
    for i=1:size(samples, 1)
        probs[i,:] = predict_probs(model, vec(samples[i,:]))
    end
    probs
end

function StatsBase.predict(model::ClassificationModel, samples::Matrix{Float64})
    [StatsBase.predict(model, vec(samples[i,:])) for i=1:size(samples,1)]
end