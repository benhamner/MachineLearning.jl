abstract SupervisedModel
abstract SupervisedModelOptions

abstract ClassificationModel <: SupervisedModel
abstract RegressionModel     <: SupervisedModel

abstract ClassificationModelOptions <: SupervisedModelOptions
abstract RegressionModelOptions     <: SupervisedModelOptions

abstract Transformer
abstract TransformerOptions

type DataFrameModel
    model::SupervisedModel
    colnames::Vector{Symbol}
end

function float_dataframe(df::DataFrame)
    res = copy(df)
    for (i,(name,col)) = enumerate(eachcol(df))
        res[name] = array(col, 0.0)*1.0
    end
    res
end

function float_matrix(df::DataFrame)
    columns = names(df)
    res = Array(Float64, (nrow(df), ncol(df)))
    for (i,(name,col)) = enumerate(eachcol(df))
        res[:,i] = array(col, 0.0)
    end
    res
end

function StatsBase.fit(df::DataFrame, target_column::Symbol, opts::SupervisedModelOptions)
    y = array(df[target_column])
    if typeof(opts) <: RegressionModelOptions
        y *= 1.0
    end
    columns = filter(x->x!=target_column, names(df))
    columns = filter(x->eltype(df[x]) <: Number, columns)
    x = float_matrix(df[columns])
    model = fit(x, y, opts)
    DataFrameModel(model, columns)
end

function predict_probs(model::ClassificationModel, samples::Matrix{Float64})
    probs = Array(Float64, size(samples, 1), length(classes(model)))
    for i=1:size(samples, 1)
        probs[i,:] = predict_probs(model, vec(samples[i,:]))
    end
    probs
end

function predict_probs(model::DataFrameModel, df::DataFrame)
    samples = float_matrix(df[model.colnames])
    predict_probs(model.model, samples)
end

function StatsBase.predict(model::ClassificationModel, samples::Matrix{Float64})
    [StatsBase.predict(model, vec(samples[i,:])) for i=1:size(samples,1)]
end

function StatsBase.predict(model::DataFrameModel, df::DataFrame)
    samples = float_matrix(df[model.colnames])
    predict(model.model, samples)
end

function StatsBase.predict(model::RegressionModel, samples::Matrix{Float64})
    [StatsBase.predict(model, vec(samples[i,:]))::Float64 for i=1:size(samples,1)]
end