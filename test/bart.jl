using Base.Test
using MachineLearning

options = bart_options()

probability_one_terminal_node    = 1 - nonterminal_node_prior(options, 1)
probability_two_terminal_nodes   = (1 - probability_one_terminal_node) * (1 - nonterminal_node_prior(options, 2))^2.0
probability_three_terminal_nodes = 2.0*(1 - probability_one_terminal_node) * (1 - nonterminal_node_prior(options, 2)) * nonterminal_node_prior(options, 2) * (1 - nonterminal_node_prior(options, 3))^2.0

@test abs(0.05-probability_one_terminal_node)/0.05 < 0.02
@test abs(0.55-probability_two_terminal_nodes)/0.55 < 0.02
@test abs(0.28-probability_three_terminal_nodes)/0.28 < 0.02

@test_throws BartTreeTransformationProbabilies(0.4, 0.3, 0.2)
# Shouldn't throw an error
transform_probabilities = BartTreeTransformationProbabilies(0.4, 0.3, 0.3)

model = randn(10)
model[3] = 100.0
x     = randn(2500, 10)
y     = x*model + randn(2500)
x_train, y_train, x_test, y_test = split_train_test(x, y)
bart = fit(x_train, y_train, bart_options())
yhat = predict(bart, x_test)
correlation = cor(y_test, yhat)
println("Bart Linear Pearson Correlation: ", correlation)
@test correlation>0.80

forest = fit(x_train, y_train, regression_forest_options())
yhat = predict(forest, x_test)
correlation = cor(y_test, yhat)
println("RF   Linear Pearson Correlation: ", correlation)
