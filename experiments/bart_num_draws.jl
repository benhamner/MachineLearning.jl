using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               bart_options(num_trees=10, num_draws=100),
               bart_options(num_trees=10, num_draws=1000)]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, options, ["100 Draws", "1000 Draws"], cor)

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
