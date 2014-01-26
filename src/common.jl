abstract SupervisedModel
abstract SupervisedModelOptions

abstract ClassificationModel <: SupervisedModel
abstract RegressionModel     <: SupervisedModel

abstract Transformer
abstract TransformerOptions

type DataFrameClassificationModel
    model::ClassificationModel
    colnames::Vector
end

function float_matrix(df::DataFrame)
    columns = names(df)
    res = Array(Float64, (nrow(df), ncol(df)))
    for (i,(name,col)) = enumerate(df)
        res[:,i] = col
    end
    res
end

function fit(df::DataFrame, target_column::String, opts::SupervisedModelOptions)
    y = [x for x=df[target_column]]
    colnames = filter(x->x!=target_column, names(df))
    x = float_matrix(df[colnames])
    model = fit(x, y, opts)
    DataFrameClassificationModel(model, colnames)
end

function predict_probs(model::ClassificationModel, samples::Matrix{Float64})
    probs = Array(Float64, size(samples, 1), length(classes(model)))
    for i=1:size(samples, 1)
        probs[i,:] = predict_probs(model, vec(samples[i,:]))
    end
    probs
end

function predict_probs(model::DataFrameClassificationModel, df::DataFrame)
    samples = float_matrix(df[model.colnames])
    predict_probs(model.model, samples)
end

function StatsBase.predict(model::ClassificationModel, samples::Matrix{Float64})
    [StatsBase.predict(model, vec(samples[i,:])) for i=1:size(samples,1)]
end

function StatsBase.predict(model::DataFrameClassificationModel, df::DataFrame)
    samples = float_matrix(df[model.colnames])
    predict(model.model, samples)
end