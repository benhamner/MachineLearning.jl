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