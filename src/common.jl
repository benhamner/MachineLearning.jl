abstract SupervisedModel
abstract SupervisedModelOptions

abstract ClassificationModel <: SupervisedModel
abstract RegressionModel     <: SupervisedModel

abstract ClassificationModelOptions <: SupervisedModelOptions
abstract RegressionModelOptions     <: SupervisedModelOptions

abstract Transformer
abstract TransformerOptions

type DataFrameModel <: SupervisedModel
    model::SupervisedModel
    colnames::Vector{Symbol}
end

abstract SupervisedDataSet
type SupervisedMatrix <: SupervisedDataSet
    x::Matrix{Float64}
    y::Vector
end
type SupervisedDataFrame <: SupervisedDataSet
    df::DataFrame
    target_column::Symbol
end

data_set_x(data::SupervisedMatrix) = data.x
data_set_y(data::SupervisedMatrix) = data.y

function data_frame_feature_columns(data::SupervisedDataFrame)
    columns = filter(x->x!=data.target_column, names(data.df))
    columns = filter(x->eltype(data.df[x]) <: Number, columns)
    columns
end

function float_dataframe(df::DataFrame)
    res = copy(df)
    for (i,(name,col)) = enumerate(eachcol(df))
        res[name] = convert(Array, col, 0.0)*1.0
    end
    res
end

function float_matrix(df::DataFrame)
    columns = names(df)
    res = Array(Float64, (nrow(df), ncol(df)))
    for (i,(name,col)) = enumerate(eachcol(df))
        if eltype(col)<:String
            array_col = convert(Array, col, "missing")
            classes = sort(unique(array_col))
            classes_map = Dict([zip(classes, 1:length(classes))...]) # TODO: cleanup post julia-0.3 compat
            res[:,i] = Float64[float(classes_map[v]) for v=array_col]
        else
            res[:,i] = convert(Array, col, 0.0)
        end
    end
    res
end

data_set_x(data::SupervisedDataFrame) = float_matrix(data.df[data_frame_feature_columns(data)])
data_set_y(data::SupervisedDataFrame) = typeof(data.df[data.target_column]) <: Array ? data.df[data.target_column] : array(data.df[data.target_column])
data_set_x_y(data::SupervisedDataSet) = data_set_x(data), data_set_y(data)

function StatsBase.fit(data::SupervisedDataFrame, opts::SupervisedModelOptions)
    x, y = data_set_x_y(data)
    columns = data_frame_feature_columns(data)
    DataFrameModel(fit(x, y, opts), columns)
end
StatsBase.fit(df::DataFrame, target_column::Symbol, opts::SupervisedModelOptions)  = fit(SupervisedDataFrame(df, target_column), opts)
StatsBase.fit(data::SupervisedMatrix, opts::SupervisedModelOptions) = fit(data.x, data.y, opts)

function predict_probs(model::ClassificationModel, x::Matrix{Float64})
    probs = Array(Float64, size(x, 1), length(classes(model)))
    for i=1:size(x, 1)
        probs[i,:] = predict_probs(model, vec(x[i,:]))
    end
    probs
end

function predict_probs(model::DataFrameModel, df::DataFrame)
    x = float_matrix(df[model.colnames])
    predict_probs(model.model, x)
end

function StatsBase.predict(model::ClassificationModel, x::Matrix{Float64})
    [StatsBase.predict(model, vec(x[i,:])) for i=1:size(x,1)]
end
function StatsBase.predict(model::RegressionModel, x::Matrix{Float64})
    [StatsBase.predict(model, vec(x[i,:]))::Float64 for i=1:size(x,1)]
end
StatsBase.predict(model::SupervisedModel, data::SupervisedMatrix) = predict(model, data.x)
StatsBase.predict(model::DataFrameModel, df::DataFrame) = predict(model.model, float_matrix(df[model.colnames]))
StatsBase.predict(model::DataFrameModel, data::SupervisedDataFrame) = predict(model, data.df)

StatsBase.sample(model::DataFrameModel, df::DataFrame) = sample(model.model, float_matrix(df[model.colnames]))
StatsBase.sample(model::DataFrameModel, data::SupervisedDataFrame) = sample(model, data.df)
