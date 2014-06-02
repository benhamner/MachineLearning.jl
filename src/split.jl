type TrainTestSplit
    x::Matrix{Float64}
    y::Vector
    train_indices::Vector{Int}
    test_indices::Vector{Int}
end

train_set(s::TrainTestSplit) = (s.x[s.train_indices,:], s.y[s.train_indices])
test_set( s::TrainTestSplit) = (s.x[s.test_indices, :], s.y[s.test_indices])
train_set_x(s::TrainTestSplit) = s.x[s.train_indices,:]
train_set_y(s::TrainTestSplit) = s.y[s.train_indices]
test_set_x( s::TrainTestSplit) = s.x[s.test_indices,:]
test_set_y( s::TrainTestSplit) = s.y[s.test_indices]

type CrossValidationSplit
    x::Matrix{Float64}
    y::Vector
    groups::Vector{Int}
end

train_set(s::CrossValidationSplit, k::Int) = (s.x[s.groups.!=k,:], s.y[s.groups.!=k])
test_set( s::CrossValidationSplit, k::Int) = (s.x[s.groups.!=k,:], s.y[s.groups.!=k])

function split_train_test(x::Matrix{Float64}, y::Vector; split_fraction::Float64=0.5, seed::Union(Int, Nothing)=Nothing())
    @assert size(x, 1)==length(y)
    @assert size(x, 1)>1
    @assert split_fraction>0.0
    @assert split_fraction<1.0

    if typeof(seed)==Int
        srand(seed)
    end

    i = shuffle([1:length(y)])
    cutoff = max(int(floor(split_fraction*length(y))), 1)
    TrainTestSplit(x, y, i[1:cutoff], i[cutoff+1:end])
end

function split_train_test(df::DataFrame; split_fraction::Float64=0.5, seed::Union(Int, Nothing)=Nothing())
    @assert nrow(df)>1
    @assert split_fraction>0.0
    @assert split_fraction<1.0

    if typeof(seed)==Int
        srand(seed)
    end

    i = shuffle([1:nrow(df)])
    cutoff = max(int(floor(split_fraction*nrow(df))), 1)
    
    train = df[i[1:cutoff],:]
    test  = df[i[cutoff+1:length(i)],:]

    train, test
end

function split_cross_valid(x::Matrix{Float64}, y::Vector; num_folds::Int=10, seed::Union(Int, Nothing)=Nothing())
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
    model = fit(train_set_x(split), train_set_y(split), opts)
    yhat = predict(model, test_set_x(split))
    metric(test_set_y(split), yhat)
end
