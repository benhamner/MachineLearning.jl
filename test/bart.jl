using Base.Test
using MachineLearning

options = bart_options()

probability_one_terminal_node    = 1 - nonterminal_node_prior(options, 1)
probability_two_terminal_nodes   = (1 - probability_one_terminal_node) * (1 - nonterminal_node_prior(options, 2))^2.0
probability_three_terminal_nodes = 2.0*(1 - probability_one_terminal_node) * (1 - nonterminal_node_prior(options, 2)) * nonterminal_node_prior(options, 2) * (1 - nonterminal_node_prior(options, 3))^2.0

@test abs(0.05-probability_one_terminal_node)/0.05 < 0.02
@test abs(0.55-probability_two_terminal_nodes)/0.55 < 0.02
@test abs(0.28-probability_three_terminal_nodes)/0.28 < 0.02

@test_throws Exception BartTreeTransformationProbabilies(0.4, 0.3, 0.2)
# Shouldn't throw an error
transform_probabilities = BartTreeTransformationProbabilies(0.4, 0.3, 0.3)

residuals  = [-0.05,0.0,0.05,0.1,0.2,0.3]
leaf_root  = BartLeaf(residuals, [1:6])
leaf_left  = BartLeaf(residuals, [1:3])
leaf_right = BartLeaf(residuals, [4:6])
branch     = DecisionBranch(1, 0.0, leaf_left, leaf_right)
leaf_params = BartLeafParameters(0.01, 0.01, 3.0, 0.001)

alpha = exp(log_likelihood(branch, leaf_params)-log_likelihood(leaf_root, leaf_params))
# TODO: analyze the effect of sigma and sigma_prior on alpha in cases like this
@test alpha>1.0
@test log_likelihood(branch, leaf_params) == log_likelihood(leaf_left, leaf_params)+log_likelihood(leaf_right, leaf_params)

m = randn(10)
m[3] = 100.0
x     = randn(2500, 10)
y     = x*m + randn(2500)
split = split_train_test(x, y)
opts = bart_options()
model = fit(train_set_x(split), train_set_y(split), opts)
yhat = predict(model, test_set_x(split))
correlation = cor(test_set_y(split), yhat)
println("Bart Linear Pearson Correlation: ", correlation)
@test correlation>0.80
@test length(model.forests)==floor(opts.num_draws/opts.thinning)

correlation = evaluate(split, regression_forest_options(), cor)
println("RF   Linear Pearson Correlation: ", correlation)
