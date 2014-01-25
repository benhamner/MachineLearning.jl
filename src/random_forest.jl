using .MachineLearning
using StatsBase

type RandomForestOptions <: SupervisedModelOptions
    num_trees::Int
    display::Bool
end

function random_forest_options(;num_trees::Int=100,
                               display::Bool=false)
    RandomForestOptions(num_trees, display)
end

type RandomForest <: ClassificationModel
    trees::Vector{DecisionTree}
    classes::Vector
    options::RandomForestOptions
end

function classes(forest::RandomForest)
    forest.classes
end

function fit(x::Matrix{Float64}, y::Vector, opts::RandomForestOptions)
    tree_opts = decision_tree_options(features_per_split_fraction=0.5)
    trees = Array(DecisionTree, 0)
    for i=1:opts.num_trees
        shuffle_locs = rand(1:size(x,1), size(x,1))
        tree = fit(x[shuffle_locs,:], y[shuffle_locs], tree_opts)
        if opts.display
            println("Tree ", i, "\tNodes: ", length(tree), "\tDepth: ", depth(tree))
        end
        push!(trees, tree)
    end
    RandomForest(trees, trees[1].classes, opts)
end

function predict_probs(forest::RandomForest, sample::Vector{Float64})
    mean([predict_probs(tree, sample) for tree=forest.trees])
end

function StatsBase.predict(forest::RandomForest, sample::Vector{Float64})
    probs = predict_probs(forest, sample)
    forest.classes[minimum(find(x->x==maximum(probs), probs))]
end

function Base.show(io::IO, forest::RandomForest)
    info = join(["Random Forest",
                 @sprintf("    %d Trees",length(forest.trees)),
                 @sprintf("    %f Nodes per tree", mean([length(tree) for tree=forest.trees])),
                 @sprintf("    %d Classes",length(forest.classes))], "\n")
    print(io, info)
end