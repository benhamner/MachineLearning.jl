type BartLeaf <: DecisionLeaf
    mu::Float64
    sigma2::Float64
    a::Float64
    b::Float64
    post_m::Float64
    post_s::Float64
    r_mean::Float64
    s2::Float64
    train_data_indices::Vector{Int}

    function BartLeaf(r::Vector{Float64}, train_data_indices::Vector{Int})
        mu     = 0.0
        sigma2 = 1.0
        a      = 1.0

        leaf_r = r[train_data_indices]
        r_mean = mean(leaf_r)
        s2     = sum((leaf_r-r_mean).^2)
        b      = length(train_data_indices) / sigma2
        post_m = b*r_mean / (a+b)
        post_s = 1.0 / sqrt(a+b)

        new(mu, sigma2, a, b, post_m, post_s, r_mean, s2, train_data_indices)
    end
end

type BartTreeTransformationProbabilies
    node_birth_death::Float64
    change_decision_rule::Float64
    swap_decision_rule::Float64

    function BartTreeTransformationProbabilies(n, c, s)
        assert(n+c+s==1.0)
        new(n, c, s)
    end
end
BartTreeTransformationProbabilies() = BartTreeTransformationProbabilies(0.5, 0.4, 0.1)

type BartOptions <: RegressionModelOptions
    num_trees::Int
    burn_in::Int
    num_draws::Int
    alpha::Float64
    beta::Float64
    k::Float64
    transform_probabilities::BartTreeTransformationProbabilies
end
BartOptions() = BartOptions(10, 200, 1000, 0.95, 2.0, BartTreeTransformationProbabilies())

function bart_options(;num_trees::Int=10,
                      burn_in::Int=200,
                      num_draws::Int=1000,
                      alpha::Float64=0.95,
                      beta::Float64=2.0,
                      k::Float64=2.0,
                      transform_probabilities::BartTreeTransformationProbabilies=BartTreeTransformationProbabilies())
    BartOptions(num_trees, burn_in, num_draws, alpha, beta, k, transform_probabilities)
end

type BartTree <: AbstractRegressionTree
    root::DecisionNode
end

# This is really a single iteration / state.
type Bart <: RegressionModel
    trees::Vector{BartTree}
    sigma::Float64
    sigma_hat::Float64
    options::BartOptions
end

function nonterminal_node_prior(alpha::Float64, beta::Float64, depth::Int)
    # using the convention that the root node has depth=1
    # BART paper implies that root node has depth=0
    alpha * depth^(-beta)
end

function nonterminal_node_prior(opts::BartOptions, depth::Int)
    nonterminal_node_prior(opts.alpha, opts.beta, depth)
end

function sigma_prior(x::Matrix{Float64}, y::Vector{Float64})
    linear_model = x\y
    sigma_hat = std(x*linear_model-y)
end

function initialize_bart(x::Matrix{Float64}, y::Vector{Float64}, opts::BartOptions)
    trees = Array(BartTree, 0)
    y_bar = mean(y)
    for i=1:opts.num_trees
        push!(trees, BartTree(BartLeaf(y_bar, [1:size(x,1)])))
    end
    sigma_hat = sigma_prior(x, y)
    Bart(trees, 1.0, sigma_hat, opts)
end

function draw_sigma(bart::Bart)
    # Default setting for sigma prior. Eventually move these settings to BartOptions
    v = 3.0
    q = 0.90
    inverse_gamma = InverseGamma(v/2.0, 1/2.0)
    lambda = bart.sigma_hat^2.0/quantile(inverse_gamma, q)/v
    sigma = sqrt(v*lambda*rand(inverse_gamma))
    sigma
end

function draw_sigma!(bart::Bart)
    sigma = draw_sigma(bart)
    bart.sigma = sigma
end

function update_tree!(tree::BartTree, opts::BartOptions, x::Matrix{Float64}, r::Vector{Float64})
    select_action = rand()
    if select_action < opts.transform_probabilities.node_birth_death
        alpha, updated = node_birth_death!(tree, x, r, opts)
    elseif select_action < opts.transform_probabilities.node_birth_death + opts.transform_probabilities.change_decision_rule
        alpha, updated = change_decision_rule!(tree, x, r)
    else
        alpha, updated = swap_decision_rule!(tree, x, r)
    end
    alpha, updated
end

function probability_node_birth(tree::BartTree)
    if typeof(tree.root) == BartLeaf
        return 1.0
    else
        return 0.5
    end
end

function birth_node(tree::BartTree)
    if typeof(tree.root) == BartLeaf
        leaf = tree.root
        leaf_node_probability = 1.0
    else
        leaf_nodes = all_leaf_nodes(tree)
        i = rand(1:length(leaf_nodes))
        leaf = leaf_nodes[i]
        leaf_probability = 1.0/length(leaf_nodes)
    end

    leaf, leaf_probability
end

function all_leaf_nodes(tree::BartTree)
    leaf_nodes = Array(BartLeaf, 0)
    all_leaf_nodes!(tree.root, leaf_nodes)
    leaf_nodes
end

function all_leaf_nodes!(branch::DecisionBranch, leaf_nodes::Vector{BartLeaf})
    all_leaf_nodes!(branch.left,  leaf_nodes)
    all_leaf_nodes!(branch.right, leaf_nodes)
end

function all_leaf_nodes!(leaf::BartLeaf, leaf_nodes::Vector{BartLeaf})
    push!(leaf_nodes, leaf)
end

function all_nog_nodes(tree::BartTree)
    nog_nodes = Array(DecisionBranch, 0)
    all_nog_nodes!(tree.root, nog_nodes)
    nog_nodes
end

function all_nog_nodes!(branch::DecisionBranch, nog_nodes::Vector{DecisionBranch})
    if typeof(branch.left)==BartLeaf && typeof(branch.right)==BartLeaf
        push!(nog_nodes, branch)
    else
        all_nog_nodes!(branch.left,  nog_nodes)
        all_nog_nodes!(branch.right, nog_nodes)
    end
end

function all_nog_nodes!(leaf::BartLeaf, nog_nodes::Vector{DecisionBranch})
    # no action
end

function depth(tree::BartTree, leaf::BartLeaf)
    depth(tree.root, leaf)
end

function depth(branch::DecisionBranch, leaf::BartLeaf)
    left_depth  = depth(branch.left)
    right_depth = depth(branch.right)
    left_depth  = left_depth > 0  ? left_depth  + 1 : 0
    right_depth = right_depth > 0 ? right_depth + 1 : 0
    max(left_depth, right_depth)
end

function depth(leaf2::BartLeaf, leaf::BartLeaf)
    leaf==leaf2 ? 1 : 0
end

function data_or_none(a, b)
    if a==None
        val = b
    else
        val = a
    end
    val
end

function parent(tree::BartTree, node::DecisionNode)
    parent(tree.root, node)
end

function parent(branch::DecisionBranch, node::DecisionNode)
    if branch.left==node || branch.right==node
        this_parent = branch
    else
        this_parent = data_or_none(parent(branch.left, node), parent(branch.right, node))
    end
    this_parent
end

function parent(leaf::BartLeaf, node::DecisionNode)
    None
end

function growth_prior(leaf::BartLeaf, leaf_depth::Int, opts::BartOptions)
    branch_prior = nonterminal_node_prior(opts, leaf_depth)
    length(leaf.train_data_indices) >= 5 ? branch_prior : 0.001*branch_prior
end

function log_likelihood(leaf::BartLeaf)
    ll = 0.0
    if length(leaf.train_data_indices)==0
        ll = -10000000.0
    else
        ll  = 0.5*log(leaf.a/(leaf.a+leaf.b))
        ll -= leaf.s2/(2.0*leaf.sigma2)
        ll -= 0.5*a*b*leaf.r_mean^2/(a+b)
    end
    ll
end

function log_likelihood(branch::DecisionBranch)
    log_likelihood(branch.left)+log_likelihood(branch.right)
end

function count_nodes_with_two_leaf_children(leaf::BartLeaf)
    0
end

function count_nodes_with_two_leaf_children(branch::DecisionBranch)
    if typeof(branch.left)==BartLeaf && typeof(branch.right)==BartLeaf
        return 1
    end
    count_nodes_with_two_leaf_children(branch.left) + count_nodes_with_two_leaf_children(branch.right)
end

function count_nodes_with_two_leaf_children(tree::BartTree)
    count_nodes_with_two_leaf_children(tree.root)
end
 
function node_birth!(tree::BartTree, x::Matrix{Float64}, r::Vector{Float64}, probability_birth::Float64, opts::BartOptions)
    leaf, leaf_node_probability = birth_node(tree)
    leaf_depth    = depth(tree, leaf)
    leaf_prior    = growth_prior(leaf, leaf_depth, opts)
    ll_before     = log_likelihood(leaf)
    split_feature = rand(1:size(x,2))
    split_loc     = rand(1:length(leaf.train_data_indices)-1)
    feature       = x[leaf.train_data_indices, split_feature]
    split_value   = sort(feature)[split_loc]
    left_indices  = leaf.train_data_indices[map(z->z<=split_value, feature)]
    right_indices = leaf.train_data_indices[map(z->z >split_value, feature)]
    left_leaf     = BartLeaf(r, left_indices)
    right_leaf    = BartLeaf(r, right_indices)
    branch        = DecisionBranch(split_feature, split_value, left_leaf, right_leaf)

    left_prior    = growth_prior(left_leaf , leaf_depth+1, opts)
    right_prior   = growth_prior(right_leaf, leaf_depth+1, opts)
    ll_after      = log_likelihood(branch)

    parent_branch = parent(tree, leaf)
    num_nog_nodes = count_nodes_with_two_leaf_children(tree)
    if parent_branch == None 
        num_nog_nodes += 1
    elseif typeof(parent_branch.left) != BartLeaf || typeof(parent_branch.right) != BartLeaf
        num_nog_nodes += 1
    end

    p_nog = 1.0/num_nog_nodes
    p_dy  = 0.5 #1.0-probability_node_birth(tree)

    alpha1 = (leaf_prior*(1.0-left_prior)*(1.0-right_prior)*p_dy*p_nog)/((1.0-leaf_prior)*probability_birth*leaf_node_probability)
    alpha  = alpha1 * exp(ll_after-ll_before)

    if rand()<alpha
        if parent_branch == None
            tree.root = branch
        else
            if leaf==parent_branch.left
                parent_branch.left  = branch
            else
                parent_branch.right = branch
            end
        end
        updated = true
    else
        updated = false
    end

    alpha, updated
end

function death_node(tree::BartTree)
    nog_nodes = all_nog_nodes(tree)
    nog_nodes[rand(1:length(nog_nodes))], 1.0/length(nog_nodes)
end

function node_death!(tree::BartTree, r::Vector{Float64}, probability_death::Float64, opts::BartOptions)
    branch, p_nog = death_node(tree)
    leaf_depth    = depth(tree, branch.left)
    left_prior    = growth_prior(branch.left, leaf_depth, opts)
    right_prior   = growth_prior(branch.left, leaf_depth, opts)
    ll_before     = log_likelihood(branch)
    leaf          = BartLeaf(r, sort(vcat(branch.left.train_data_indices, branch.right.train_data_indices)))
    ll_after      = log_likelihood(leaf)

    parent_branch = parent(tree, branch)
    if parent_branch == None
        probability_birth_after = 1.0
    else
        probability_birth_after = 0.5
    end
    prior_grow = growth_prior(leaf, leaf_depth-1, opts)
    probability_birth_leaf = 1.0 / (length(all_leaf_nodes(tree))-1)

    alpha1 = ((1.0-prior_grow)*probability_birth_after*probability_birth_leaf)/(prior_grow*(1.0-left_prior)*(1.0-right_prior)*probability_death*p_nog)
    alpha  = alpha1*exp(ll_after-ll_before)

    if rand()<alpha
        if parent_branch == None 
            tree.root = leaf
        else
            if parent_branch.left == branch
                parent_branch.left =  leaf
            else
                parent_branch.right = leaf
            end
        end
        updated = true
    else
        updated = false
    end
end

function node_birth_death!(tree::BartTree, x::Matrix{Float64}, r::Vector{Float64}, opts::BartOptions)
    probability_birth = probability_node_birth(tree)
    if rand() < probability_birth
        alpha, updated = node_birth!(tree, x, r, probability_birth, opts)
    else
        probability_death = 1.0 - probability_birth
        alpha, updated = node_death!(tree, r, probability_death, opts)
    end
    alpha, updated
end

function change_decision_rule!(tree::BartTree, x::Matrix{Float64}, r::Vector{Float64})
    error("Not implemented yet")
    alpha, updated
end

function swap_decision_rule!(tree::BartTree, x::Matrix{Float64}, r::Vector{Float64})
    error("Not implemented yet")
    alpha, updated
end

function posterior_mu_sigma(prior_mu, a, sigma_hat, y_bar, num_observations)
    b = num_observations / sigma_hat^2
    posterior_mu = b*y_bar/(a+b)
    posterior_sigma = 1 / sqrt(a+b)
    posterior_mu, posterior_sigma
end

function fit_predict(x_train::Matrix{Float64}, y_train::Vector{Float64}, opts::BartOptions, x_test::Matrix{Float64})
    bart = initialize_bart(x_train, y_train, opts)
    for i=1:opts.num_draws
        draw_sigma!(bart)
        y_hat = predict(bart, x_train)
        for i=1:opts.num_trees
            residuals = y_train-y_hat+predict(bart.trees[i], x_train)
            update_tree!(bart.trees[i], bart.options, x_train, residuals)
        end
        if i>opts.burn_in
            # store predictions
        end
    end
end

function StatsBase.predict(bart::Bart, sample::Vector{Float64})
    sum([predict(tree, sample) for tree=bart.trees])
end