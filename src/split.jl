abstract TrainTestSplit
type MatrixTrainTestSplit <: TrainTestSplit
    x::Matrix{Float64}
    y::Vector
    train_indices::Vector{Int}
    test_indices::Vector{Int}
end
type DataFrameTrainTestSplit <: TrainTestSplit
    df::DataFrame
    target_column::Symbol
    train_indices::Vector{Int}
    test_indices::Vector{Int}
end

train_set(s::MatrixTrainTestSplit) = SupervisedMatrix(s.x[s.train_indices,:], s.y[s.train_indices])
test_set( s::MatrixTrainTestSplit) = SupervisedMatrix(s.x[s.test_indices, :], s.y[s.test_indices])
train_set(s::DataFrameTrainTestSplit) = SupervisedDataFrame(s.df[s.train_indices,:], s.target_column)
test_set( s::DataFrameTrainTestSplit) = SupervisedDataFrame(s.df[s.test_indices, :], s.target_column)
train_set_x(s::TrainTestSplit) = data_set_x(train_set(s))
train_set_y(s::TrainTestSplit) = data_set_y(train_set(s))
test_set_x( s::TrainTestSplit) = data_set_x(test_set(s))
test_set_y( s::TrainTestSplit) = data_set_y(test_set(s))
train_set_x_y(s::TrainTestSplit) = train_set_x(s), train_set_y(s)
test_set_x_y( s::TrainTestSplit) = test_set_x(s), test_set_y(s)

StatsBase.fit(split::TrainTestSplit, opts::SupervisedModelOptions) = fit(train_set(split), opts)
StatsBase.predict(model::SupervisedModel, split::TrainTestSplit)   = predict(model, test_set(split))
StatsBase.sample(model::SupervisedModel, split::TrainTestSplit)    = sample(model, test_set(split))

type CrossValidationSplit
    x::Matrix{Float64}
    y::Vector
    groups::Vector{Int}
end

train_set(s::CrossValidationSplit, k::Int) = (s.x[s.groups.!=k,:], s.y[s.groups.!=k])
test_set( s::CrossValidationSplit, k::Int) = (s.x[s.groups.==k,:], s.y[s.groups.==k])

function split_train_test(x::Matrix{Float64}, y::Vector; split_fraction::Float64=0.5, seed::Union(Int, Nothing)=nothing)
    @assert size(x, 1)==length(y)
    @assert size(x, 1)>1
    @assert split_fraction>0.0
    @assert split_fraction<1.0

    if typeof(seed)==Int
        srand(seed)
    end

    i = shuffle([1:length(y)])
    cutoff = max(int(floor(split_fraction*length(y))), 1)
    MatrixTrainTestSplit(x, y, i[1:cutoff], i[cutoff+1:end])
end

function split_train_test(df::DataFrame, target_column::Symbol; split_fraction::Float64=0.5, seed::Union(Int, Nothing)=nothing)
    @assert nrow(df)>1
    @assert split_fraction>0.0
    @assert split_fraction<1.0

    if typeof(seed)==Int
        srand(seed)
    end

    i = shuffle([1:nrow(df)])
    cutoff = max(int(floor(split_fraction*nrow(df))), 1)
    
    DataFrameTrainTestSplit(df, target_column, i[1:cutoff], i[cutoff+1:length(i)])
end
split_train_test(data::SupervisedDataFrame; split_fraction::Float64=0.5, seed::Union(Int, Nothing)=nothing) = split_train_test(data.df, data.target_column, split_fraction=split_fraction, seed=seed)

function split_cross_valid(x::Matrix{Float64}, y::Vector; num_folds::Int=10, seed::Union(Int, Nothing)=nothing)
    @assert size(x, 1)==length(y)
    @assert size(x, 1)>=num_folds
    @assert num_folds>1

    if typeof(seed)==Int
        srand(seed)
    end

    i = shuffle([1:length(y)])
    fold_size = int(floor(length(y)/num_folds))
    remainder = length(y)-num_folds*fold_size
    groups = zeros(Int, length(y))
    cursor = 1
    group = 1
    while cursor<=length(y)
        this_fold_size = group <= remainder ? fold_size+1:fold_size
        groups[i[cursor:cursor+this_fold_size-1]] = group
        group += 1
        cursor += this_fold_size
    end
    CrossValidationSplit(x, y, groups)
end

function evaluate(split::TrainTestSplit, opts::SupervisedModelOptions, metric::Function)
    model = fit(train_set(split), opts)
    yhat = predict(model, test_set(split))
    metric(test_set_y(split), yhat)
end

function evaluate(split::CrossValidationSplit, opts::SupervisedModelOptions, metric::Function)
    yhat = [split.y[1] for i=1:length(split.y)]
    for i=unique(split.groups)
        x_train, y_train = train_set(split, i)
        x_test,  y_test  = test_set( split, i)
        model = fit(x_train, y_train, opts)
        yhat[split.groups.==i] = predict(model, x_test)
    end
    metric(split.y, yhat)
end
