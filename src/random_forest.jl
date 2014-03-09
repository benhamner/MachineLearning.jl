type ClassificationForestOptions <: ClassificationModelOptions
    num_trees::Int
    display::Bool
end

function classification_forest_options(;num_trees::Int=100,
                                       display::Bool=false)
    ClassificationForestOptions(num_trees, display)
end

type RegressionForestOptions <: RegressionModelOptions
    num_trees::Int
    display::Bool
end

function regression_forest_options(;num_trees::Int=100,
                                   display::Bool=false)
    RegressionForestOptions(num_trees, display)
end

type ClassificationForest <: ClassificationModel
    trees::Vector{ClassificationTree}
    classes::Vector
    options::ClassificationForestOptions
end

type RegressionForest <: RegressionModel
    trees::Vector{RegressionTree}
    options::RegressionForestOptions
end

function classes(forest::ClassificationForest)
    forest.classes
end

function fit(x::Matrix{Float64}, y::Vector, opts::ClassificationForestOptions)
    tree_opts = classification_tree_options(features_per_split_fraction=0.5)
    trees = Array(ClassificationTree, 0)
    for i=1:opts.num_trees
        shuffle_locs = rand(1:size(x,1), size(x,1))
        tree = fit(x[shuffle_locs,:], y[shuffle_locs], tree_opts)
        if opts.display
            println("Tree ", i, "\tNodes: ", length(tree), "\tDepth: ", depth(tree))
        end
        push!(trees, tree)
    end
    ClassificationForest(trees, trees[1].classes, opts)
end

function fit(x::Matrix{Float64}, y::Vector{Float64}, opts::RegressionForestOptions)
    tree_opts = regression_tree_options(features_per_split_fraction=0.5, minimum_split_size = max(2, int(size(x,1)/128)))
    trees = Array(RegressionTree, 0)
    for i=1:opts.num_trees
        shuffle_locs = rand(1:size(x,1), size(x,1))
        tree = fit(x[shuffle_locs,:], y[shuffle_locs], tree_opts)
        if opts.display
            println("Tree ", i, "\tNodes: ", length(tree), "\tDepth: ", depth(tree))
        end
        push!(trees, tree)
    end
    RegressionForest(trees, opts)
end

function predict_probs(forest::ClassificationForest, sample::Vector{Float64})
    mean([predict_probs(tree, sample) for tree=forest.trees])
end

function StatsBase.predict(forest::ClassificationForest, sample::Vector{Float64})
    probs = predict_probs(forest, sample)
    forest.classes[minimum(find(x->x==maximum(probs), probs))]
end

function StatsBase.predict(forest::RegressionForest, sample::Vector{Float64})
    mean([predict(tree, sample) for tree=forest.trees])
end

function Base.show(io::IO, forest::ClassificationForest)
    info = join(["Classification Forest",
                 @sprintf("    %d Trees",length(forest.trees)),
                 @sprintf("    %f Nodes per tree", mean([length(tree) for tree=forest.trees])),
                 @sprintf("    %d Classes",length(forest.classes))], "\n")
    print(io, info)
end

function Base.show(io::IO, forest::RegressionForest)
    info = join(["Regression Forest",
                 @sprintf("    %d Trees",length(forest.trees)),
                 @sprintf("    %f Nodes per tree", mean([length(tree) for tree=forest.trees])),], "\n")
    print(io, info)
end