using Gadfly
using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               RegressionPipelineOptions(TransformerOptions[ZmuvOptions()], regression_net_options()),
               regression_forest_options(num_trees=2),
               regression_forest_options(num_trees=10),
               bart_options(num_trees=2),
               bart_options(num_trees=10)]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed+1)

res = compare(data_generator, options, ["Net", "Tree", "Forest2", "Forest10", "Forest100", "Forest200",
                                        "BART2", "BART10", "BART100", "BART200"], cor)

draw(PNG("regression_comparison.png", 8inch, 6inch), plot(res, x=:Name, y=:Score, Geom.boxplot))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
