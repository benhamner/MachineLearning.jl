using .MachineLearning
using StatsBase

abstract DecisionNode

type DecisionTreeOptions <: SupervisedModelOptions
    features_per_split_fraction::Float64
    minimum_split_size::Int
end
DecisionTreeOptions() = DecisionTreeOptions(1.0, 2)

function decision_tree_options(;features_per_split_fraction::Float64=1.0,
                               minimum_split_size::Int=2)
    DecisionTreeOptions(features_per_split_fraction,
                        minimum_split_size)
end

type DecisionLeaf <: DecisionNode
    probs::Vector{Float64}
end

type DecisionBranch <: DecisionNode
    feature::Int
    value::Float64
    left::DecisionNode
    right::DecisionNode
end

type DecisionTree <: ClassificationModel
    root::DecisionNode
    classes::Vector
    features_per_split::Int
    options::DecisionTreeOptions
end

function classes(tree::DecisionTree)
    tree.classes
end

function fit(x::Matrix{Float64}, y::Vector, opts::DecisionTreeOptions)
    classes = sort(unique(y))
    classes_map = Dict(classes, 1:length(classes))
    y_mapped = [classes_map[v]::Int for v=y]
    features_per_split = int(opts.features_per_split_fraction*size(x,2))
    features_per_split = max(1, size(x,2))
    root = train_branch(x, y_mapped, opts, length(classes), features_per_split)
    DecisionTree(root, classes, features_per_split, opts)
end

function train_branch(x::Matrix{Float64}, y::Vector{Int}, opts::DecisionTreeOptions, num_classes::Int, features_per_split::Int)
    if length(y)<opts.minimum_split_size || length(unique(y))==1
        probs = zeros(num_classes)
        for i=1:length(y)
            probs[y[i]] += 1.0/length(y)
        end
        return DecisionLeaf(probs)
    end

    score        = Inf
    best_feature = 1
    split_loc    = 1
    for feature = shuffle([1:size(x,2)])[1:features_per_split]
        i_sorted = sortperm(x[:,feature])
        g, loc = split_location(y[i_sorted], num_classes)
        if g<score 
            score        = g
            best_feature = feature
            split_loc    = loc
        end
    end
    i_sorted    = sortperm(x[:,best_feature])
    left_locs   = i_sorted[1:split_loc]
    right_locs  = i_sorted[split_loc+1:length(i_sorted)]
    left        = train_branch(x[left_locs, :], y[left_locs],  opts, num_classes, features_per_split)
    right       = train_branch(x[right_locs,:], y[right_locs], opts, num_classes, features_per_split)
    split_value = x[i_sorted[split_loc], best_feature]
    DecisionBranch(best_feature, split_value, left, right)
end

function split_location(y::Vector{Int}, num_classes::Int)
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

function predict_probs(tree::DecisionTree, sample::Vector{Float64})
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

function StatsBase.predict(tree::DecisionTree, sample::Vector{Float64})
    probs = predict_probs(tree, sample)
    tree.classes[minimum(find(x->x==maximum(probs), probs))]
end

function Base.length(tree::DecisionTree)
    length(tree.root)
end

function Base.length(branch::DecisionBranch)
    return 1+length(branch.left)+length(branch.right)
end

function Base.length(leaf::DecisionLeaf)
    return 1
end

function Base.show(io::IO, tree::DecisionTree)
    info = join(["Decision Tree",
                 @sprintf("    %d Nodes, %d Nodes Deep",length(tree), depth(tree)),
                 @sprintf("    %d Classes",length(tree.classes))], "\n")
    print(io, info)
end

function depth(tree::DecisionTree)
    depth(tree.root)
end

function depth(branch::DecisionBranch)
    return 1+max(depth(branch.left),depth(branch.right))
end

function depth(leaf::DecisionLeaf)
    return 1
end