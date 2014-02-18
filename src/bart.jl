type BartTreeTransformationProbabilies
    grow_terminal_node::Float64
    prune_terminal_node_pair::Float64
    change_decision_rule::Float64
    swap_decision_rule::Float64
end
BartTreeTransformationProbabilies() = BartTreeTransformationProbabilies(0.25, 0.25, 0.4, 0.1)

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

type BartTree
    root::DecisionNode
end

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
    for i=1:opts.num_trees
        push!(trees, BartTree(RegressionLeaf(0.0)))
    end
    sigma_hat = sigma_prior(x, y),
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

function fit_predict(x_train::Matrix{Float64}, y_train::Vector{Float64}, opts::BartOptions, x_test::Matrix{Float64})
    bart = initialize_bart(x_train, y_train, opts)
end