type ClassificationForestOptions <: ClassificationModelOptions
    num_trees::Int
    classes::Union(Vector, Nothing)
    tree_options::ClassificationTreeOptions
    display::Bool
end

function classification_forest_options(;num_trees::Int=100,
                                       classes::Union(Vector, Nothing)=nothing,
                                       tree_options::ClassificationTreeOptions=classification_tree_options(features_per_split_fraction=0.5, minimum_split_size=2),
                                       display::Bool=false)
    ClassificationForestOptions(num_trees, classes, tree_options, display)
end

type RegressionForestOptions <: RegressionModelOptions
    num_trees::Int
    tree_options::RegressionTreeOptions
    display::Bool
end

function regression_forest_options(;num_trees::Int=100,
                                   tree_options::RegressionTreeOptions=regression_tree_options(features_per_split_fraction=0.5, minimum_split_size=5),
                                   display::Bool=false)
    RegressionForestOptions(num_trees, tree_options, display)
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

function StatsBase.fit(x::Matrix{Float64}, y::Vector, opts::ClassificationForestOptions)
    classes = typeof(opts.classes)<:Nothing ? sort(unique(y)) : opts.classes
    tree_opts = opts.tree_options
    tree_opts.classes = classes
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

function StatsBase.fit(x::Matrix{Float64}, y::Vector{Float64}, opts::RegressionForestOptions)
    trees = Array(RegressionTree, 0)
    for i=1:opts.num_trees
        shuffle_locs = rand(1:size(x,1), size(x,1))
        tree = fit(x[shuffle_locs,:], y[shuffle_locs], opts.tree_options)
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
