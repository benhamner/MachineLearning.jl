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

abstract SupervisedLearningDataSet
type MatrixSupervisedLearningDataSet <: SupervisedLearningDataSet
    x::Matrix{Float64}
    y::Vector
end
type DataFrameSupervisedLearningDataSet <: SupervisedLearningDataSet
    df::DataFrame
    target_column::Symbol
end

data_set_x(data::MatrixSupervisedLearningDataSet) = data.x
data_set_y(data::MatrixSupervisedLearningDataSet) = data.y

function data_frame_feature_columns(data::DataFrameSupervisedLearningDataSet)
    columns = filter(x->x!=data.target_column, names(data.df))
    columns = filter(x->eltype(data.df[x]) <: Number, columns)
    columns
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
        if eltype(col)<:String
            array_col = array(col, "missing")
            classes = sort(unique(array_col))
            classes_map = Dict(classes, 1:length(classes))
            res[:,i] = Float64[float(classes_map[v]) for v=array_col]
        else
            res[:,i] = array(col, 0.0)
        end
    end
    res
end

data_set_x(data::DataFrameSupervisedLearningDataSet) = float_matrix(data.df[data_frame_feature_columns(data)])
function data_set_y(data::DataFrameSupervisedLearningDataSet)
    if eltype(data.df[data.target_column])<:Float64 
        return array(data.df[data.target_column], 0.0)
    end
    return [y for y=data.df[data.target_column]]
end
data_set_x_y(data::SupervisedLearningDataSet) = data_set_x(data), data_set_y(data)

function StatsBase.fit(data::DataFrameSupervisedLearningDataSet, opts::SupervisedModelOptions)
    x, y = data_set_x_y(data)
    if typeof(opts) <: RegressionModelOptions
        y *= 1.0
    end
    columns = data_frame_feature_columns(data)
    DataFrameModel(fit(x, y, opts), columns)
end
StatsBase.fit(df::DataFrame, target_column::Symbol, opts::SupervisedModelOptions)  = fit(DataFrameSupervisedLearningDataSet(df, target_column), opts)
StatsBase.fit(data::MatrixSupervisedLearningDataSet, opts::SupervisedModelOptions) = fit(data.x, data.y, opts)

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
function StatsBase.predict(model::RegressionModel, samples::Matrix{Float64})
    [StatsBase.predict(model, vec(samples[i,:]))::Float64 for i=1:size(samples,1)]
end
StatsBase.predict(model::SupervisedModel, data::MatrixSupervisedLearningDataSet) = predict(model, data.x)

function StatsBase.predict(model::DataFrameModel, df::DataFrame)
    samples = float_matrix(df[model.colnames])
    predict(model.model, samples)
end
StatsBase.predict(model::DataFrameModel, data::DataFrameSupervisedLearningDataSet) = predict(model, data.df)
