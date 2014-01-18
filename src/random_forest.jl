using MachineLearning

type RandomForestOptions
    num_trees::Int
end

function random_forest_options(;num_trees::Int=100)
    RandomForestOptions(num_trees)
end

type RandomForest
    trees::Vector{DecisionTree}
    classes::Vector
    options::RandomForestOptions
end

function train(x::Array{Float64,2}, y::Vector, opts::RandomForestOptions)
    tree_opts = decision_tree_options(features_per_split_fraction=0.5)
    trees = Array(DecisionTree, 0)
    for i=1:opts.num_trees
        shuffle_locs = rand(1:size(x,1), size(x,1))
        tree = train(x[shuffle_locs,:], y[shuffle_locs], tree_opts)
        push!(trees, tree)
    end
    RandomForest(trees, trees[1].classes, opts)
end

function predict_probs(forest::RandomForest, sample::Vector{Float64})
    mean([predict_probs(tree, sample) for tree=forest.trees])
end

function predict_probs(forest::RandomForest, samples::Array{Float64, 2})
    probs = Array(Float64, size(samples, 1), length(forest.classes))
    for i=1:size(samples, 1)
        probs[i,:] = predict_probs(forest, vec(samples[i,:]))
    end
    probs
end

function predict(forest::RandomForest, sample::Vector{Float64})
    probs = predict_probs(forest, sample)
    forest.classes[minimum(find(x->x==maximum(probs), probs))]
end

function predict(forest::RandomForest, samples::Array{Float64, 2})
    [predict(forest, vec(samples[i,:])) for i=1:size(samples,1)]
end