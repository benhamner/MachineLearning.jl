abstract Decision
typealias DecisionNode   Node{Decision}
typealias DecisionLeaf   Leaf{Decision}
type DecisionTree <: Tree{Decision}
    root::DecisionNode
end

type ClassificationTreeOptions <: ClassificationModelOptions
    features_per_split_fraction::Float64
    minimum_split_size::Int
    classes::Union(Vector, Nothing)
end
ClassificationTreeOptions() = ClassificationTreeOptions(1.0, 2, nothing)

function classification_tree_options(;features_per_split_fraction::Float64=1.0,
                               minimum_split_size::Int=2,
                               classes::Union(Vector,Nothing)=nothing)
    ClassificationTreeOptions(features_per_split_fraction,
                        minimum_split_size,
                        classes)
end

type RegressionTreeOptions <: RegressionModelOptions
    features_per_split_fraction::Float64
    minimum_split_size::Int
end
RegressionTreeOptions() = RegressionTreeOptions(1.0, 2)

function regression_tree_options(;features_per_split_fraction::Float64=1.0,
                               minimum_split_size::Int=5)
    RegressionTreeOptions(features_per_split_fraction,
                        minimum_split_size)
end

type ClassificationLeaf <: DecisionLeaf
    probs::Vector{Float64}
end

type RegressionLeaf <: DecisionLeaf
    value::Float64
end

type DecisionBranch <: Branch{Decision}
    feature::Int
    value::Float64
    left::DecisionNode
    right::DecisionNode
end

type ClassificationTree <: ClassificationModel
    tree::DecisionTree
    classes::Vector
    features_per_split::Int
    options::ClassificationTreeOptions
end
@make_tree_type(ClassificationTree, DecisionNode)

abstract AbstractRegressionTree <: RegressionModel
type RegressionTree <:  AbstractRegressionTree
    tree::DecisionTree
    features_per_split::Int
    options::RegressionTreeOptions
end
@make_tree_type(AbstractRegressionTree, DecisionNode)

function classes(tree::ClassificationTree)
    tree.classes
end

function StatsBase.fit(x::Matrix{Float64}, y::Vector, opts::ClassificationTreeOptions)
    classes = typeof(opts.classes)<:Nothing ? sort(unique(y)) : opts.classes
    classes_map = Dict([zip(classes, 1:length(classes))...]) # TODO: cleanup post julia-0.3 compat
    y_mapped = [classes_map[v]::Int for v=y]
    features_per_split = int(opts.features_per_split_fraction*size(x,2))
    features_per_split = max(1, size(x,2))
    root = train_classification_branch(x, y_mapped, opts, length(classes), features_per_split)
    ClassificationTree(DecisionTree(root), classes, features_per_split, opts)
end

function StatsBase.fit(x::Matrix{Float64}, y::Vector{Float64}, opts::RegressionTreeOptions)
    features_per_split = int(opts.features_per_split_fraction*size(x,2))
    features_per_split = max(1, size(x,2))
    root = train_regression_branch(x, y, opts, features_per_split)
    RegressionTree(DecisionTree(root), features_per_split, opts)
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
        return RegressionLeaf(mean(y))
    end

    score        = Inf
    best_feature = 1
    split_loc    = 1
    for feature = shuffle([1:size(x,2)])[1:features_per_split]
        i_sorted = sortperm(x[:,feature])
        g, loc = regression_split_location(y[i_sorted])
        if g<score 
            score        = g
            best_feature = feature
            split_loc    = loc
        end
    end
    i_sorted    = sortperm(x[:,best_feature])
    left_locs   = i_sorted[1:split_loc]
    right_locs  = i_sorted[split_loc+1:length(i_sorted)]
    left        = train_regression_branch(x[left_locs, :], y[left_locs],  opts, features_per_split)
    right       = train_regression_branch(x[right_locs,:], y[right_locs], opts, features_per_split)
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

function streaming_mse(x_sum::Float64, x_squared_sum::Float64, n::Int)
    x_bar = x_sum/n
    x_bar^2 + x_squared_sum/n - 2*x_bar*x_sum/n
end

function regression_split_location(y::Vector{Float64})
    sum_left  = 0.0
    sum_right = sum(y)
    squared_sum_left  = 0.0
    squared_sum_right = sum(y.^2)

    loc   = 1
    score = Inf
    for i=1:length(y)-1
        sum_left  += y[i]
        sum_right -= y[i]
        squared_sum_left  += y[i]^2
        squared_sum_right -= y[i]^2

        mse = streaming_mse(sum_left,  squared_sum_left,  i) +
              streaming_mse(sum_right, squared_sum_right, length(y)-i)
        if mse<score
            score = mse
            loc   = i
        end
    end
    score, loc
end

function gini(counts::Vector{Float64})
    1-sum((counts/sum(counts)).^2)
end

function predict_probs(tree::ClassificationTree, sample::Vector{Float64})
    node = tree.tree.root
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

function StatsBase.predict(tree::AbstractRegressionTree, sample::Vector{Float64})
    node = tree.tree.root
    while typeof(node)==DecisionBranch
        if sample[node.feature]<=node.value
            node=node.left
        else
            node=node.right
        end
    end
    node.value
end

function Base.length(tree::ClassificationTree)
    length(tree.tree)
end

function Base.length(tree::AbstractRegressionTree)
    length(tree.tree)
end

function Base.show(io::IO, tree::ClassificationTree)
    info = join(["Classification Tree",
                 @sprintf("    %d Nodes, %d Nodes Deep",length(tree), depth(tree)),
                 @sprintf("    %d Classes",length(tree.classes))], "\n")
    print(io, info)
end

Base.show(io::IO, branch::DecisionBranch) = print(io, pretty_string(branch, 1))
pretty_string(leaf::DecisionLeaf, indent::Integer) = string(leaf)[1:min(end, 16)]=="MachineLearning." ? string(leaf)[17:end] : string(leaf)
function pretty_string(branch::DecisionBranch, indent::Integer) 
    @sprintf("DecisionBranch(%d, %f, \n%s%s \n%s%s)", 
    branch.feature,
    branch.value,
    repeat("|", indent), 
    pretty_string(branch.left, indent+1), 
    repeat("|", indent), 
    pretty_string(branch.right, indent+1))
end
pretty_string(branch::DecisionBranch) = pretty_string(branch::DecisionBranch, 1)
