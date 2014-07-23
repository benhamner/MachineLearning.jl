using Gadfly
using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               classification_net_options(learning_rate=10.0),
               classification_net_options(stop_criteria=StopAfterIteration(1000), learning_rate=1.0),
               classification_tree_options(),
               classification_forest_options(num_trees=10),
               classification_forest_options(num_trees=100)]

data_generator(seed) = split_train_test(dataset("ggplot2", "midwest"), :Category, seed=seed)

res = compare(data_generator, options, ["NeuralNet100Iter", "NeuralNet1000Iter", "Tree", "Forest10", "Forest100"], accuracy)

draw(PNG("classification_comparison.png", 8inch, 6inch), plot(res, x=:Name, y=:Score, Geom.boxplot))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
