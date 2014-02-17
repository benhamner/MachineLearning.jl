abstract DecisionNode

type ClassificationTreeOptions <: SupervisedModelOptions
    features_per_split_fraction::Float64
    minimum_split_size::Int
end
ClassificationTreeOptions() = ClassificationTreeOptions(1.0, 2)

function classification_tree_options(;features_per_split_fraction::Float64=1.0,
                               minimum_split_size::Int=2)
    ClassificationTreeOptions(features_per_split_fraction,
                        minimum_split_size)
end

type RegressionTreeOptions <: SupervisedModelOptions
    features_per_split_fraction::Float64
    minimum_split_size::Int
end
RegressionTreeOptions() = RegressionTreeOptions(1.0, 2)

function Regression_tree_options(;features_per_split_fraction::Float64=1.0,
                               minimum_split_size::Int=2)
    RegressionTreeOptions(features_per_split_fraction,
                        minimum_split_size)
end

type ClassificationLeaf <: DecisionNode
    probs::Vector{Float64}
end

type RegressionLeaf <: DecisionNode
    value::Float64
end

type DecisionBranch <: DecisionNode
    feature::Int
    value::Float64
    left::DecisionNode
    right::DecisionNode
end

type ClassificationTree <: ClassificationModel
    root::DecisionNode
    classes::Vector
    features_per_split::Int
    options::ClassificationTreeOptions
end

type RegressionTree <: ClassificationModel
    root::DecisionNode
    features_per_split::Int
    options::RegressionTreeOptions
end

function classes(tree::ClassificationTree)
    tree.classes
end

function fit(x::Matrix{Float64}, y::Vector, opts::ClassificationTreeOptions)
    classes = sort(unique(y))
    classes_map = Dict(classes, 1:length(classes))
    y_mapped = [classes_map[v]::Int for v=y]
    features_per_split = int(opts.features_per_split_fraction*size(x,2))
    features_per_split = max(1, size(x,2))
    root = train_classification_branch(x, y_mapped, opts, length(classes), features_per_split)
    ClassificationTree(root, classes, features_per_split, opts)
end

function fit(x::Matrix{Float64}, y::Vector{Float64}, opts::RegressionTreeOptions)
    features_per_split = int(opts.features_per_split_fraction*size(x,2))
    features_per_split = max(1, size(x,2))
    root = train_regression_branch(x, y, opts, features_per_split)
    RegressionTree(root, features_per_split, opts)
end

function train_classification_branch(x::Matrix{Float64}, y::Vector{Int}, opts::ClassificationTreeOptions, num_classes::Int, features_per_split::Int)
    if length(y)<opts.minimum_split_size || length(unique(y))==1
        probs = zeros(num_classes)
        for i=1:length(y)
            probs[y[i]] += 1.0/length(y)
        end
        return ClassificationLeaf(probs)
    end

    score        = Inf
    best_feature = 1
    split_loc    = 1
    for feature = shuffle([1:size(x,2)])[1:features_per_split]
        i_sorted = sortperm(x[:,feature])
        g, loc = classification_split_location(y[i_sorted], num_classes)
        if g<score 
            score        = g
            best_feature = feature
            split_loc    = loc
        end
    end
    i_sorted    = sortperm(x[:,best_feature])
    left_locs   = i_sorted[1:split_loc]
    right_locs  = i_sorted[split_loc+1:length(i_sorted)]
    left        = train_classification_branch(x[left_locs, :], y[left_locs],  opts, num_classes, features_per_split)
    right       = train_classification_branch(x[right_locs,:], y[right_locs], opts, num_classes, features_per_split)
    split_value = x[i_sorted[split_loc], best_feature]
    DecisionBranch(best_feature, split_value, left, right)
end

function train_regression_branch(x::Matrix{Float64}, y::Vector{Float64}, opts::RegressionTreeOptions, features_per_split::Int)
    if length(y)<opts.minimum_split_size
        return ClassificationLeaf(mean(y))
    end

    score        = Inf
    best_feature = 1
    split_loc    = 1
    for feature = shuffle([1:size(x,2)])[1:features_per_split]
        i_sorted = sortperm(x[:,feature])
        g, loc = regression_split_location(y[i_sorted], num_classes)
        if g<score 
            score        = g
            best_feature = feature
            split_loc    = loc
        end
    end
    i_sorted    = sortperm(x[:,best_feature])
    left_locs   = i_sorted[1:split_loc]
    right_locs  = i_sorted[split_loc+1:length(i_sorted)]
    left        = train_branch(x[left_locs, :], y[left_locs],  opts, features_per_split)
    right       = train_branch(x[right_locs,:], y[right_locs], opts, features_per_split)
    split_value = x[i_sorted[split_loc], best_feature]
    DecisionBranch(best_feature, split_value, left, right)
end

function classification_split_location(y::Vector{Int}, num_classes::Int)
    counts_left  = zeros(num_classes)
    counts_right = zeros(num_classes)
    for i=1:length(y)
        counts_right[y[i]]+=1
    end
    loc   = 1
    score = Inf
    for i=1:length(y)-1
        counts_left[y[i]]+=1
        counts_right[y[i]]-=1
        g = i/length(y)*gini(counts_left)+(length(y)-i)/length(y)*gini(counts_right)
        if g<score
            score = g
            loc   = i
        end
    end
    score, loc
end

# TODO:: finish this function
function regression_split_location(y::Vector{Int})
    counts_left  = zeros(num_classes)
    counts_right = zeros(num_classes)
    for i=1:length(y)
        counts_right[y[i]]+=1
    end
    loc   = 1
    score = Inf
    for i=1:length(y)-1
        counts_left[y[i]]+=1
        counts_right[y[i]]-=1
        g = i/length(y)*gini(counts_left)+(length(y)-i)/length(y)*gini(counts_right)
        if g<score
            score = g
            loc   = i
        end
    end
    score, loc
end

function gini(counts::Vector{Float64})
    1-sum((counts/sum(counts)).^2)
end

function predict_probs(tree::ClassificationTree, sample::Vector{Float64})
    node = tree.root
    while typeof(node)==DecisionBranch
        if sample[node.feature]<=node.value
            node=node.left
        else
            node=node.right
        end
    end
    node.probs
end

function StatsBase.predict(tree::ClassificationTree, sample::Vector{Float64})
    probs = predict_probs(tree, sample)
    tree.classes[minimum(find(x->x==maximum(probs), probs))]
end

function Base.length(tree::ClassificationTree)
    length(tree.root)
end

function Base.length(branch::DecisionBranch)
    return 1+length(branch.left)+length(branch.right)
end

function Base.length(leaf::ClassificationLeaf)
    return 1
end

function Base.show(io::IO, tree::ClassificationTree)
    info = join(["Classification Tree",
                 @sprintf("    %d Nodes, %d Nodes Deep",length(tree), depth(tree)),
                 @sprintf("    %d Classes",length(tree.classes))], "\n")
    print(io, info)
end

function depth(tree::ClassificationTree)
    depth(tree.root)
end

function depth(branch::DecisionBranch)
    return 1+max(depth(branch.left),depth(branch.right))
end

function depth(leaf::ClassificationLeaf)
    return 1
end