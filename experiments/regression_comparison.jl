using Gadfly
using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               regression_tree_options(),
               regression_forest_options()]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, options, ["Tree", "Forest"], cor)

draw(PNG("regression_comparison.png", 8inch, 6inch), plot(res, x=:Name, y=:Score, Geom.boxplot))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
