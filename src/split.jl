type TrainTestSplit
    x::Matrix{Float64}
    y::Vector
    train_indices::Vector{Int}
    test_indices::Vector{Int}
end

train_set(s::TrainTestSplit) = (s.x[s.train_indices,:], s.y[s.train_indices])
test_set( s::TrainTestSplit) = (s.x[s.test_indices, :], s.y[s.test_indices])

type CrossValidationSplit
    x::Matrix{Float64}
    y::Vector
    groups::Vector{Int}
end

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