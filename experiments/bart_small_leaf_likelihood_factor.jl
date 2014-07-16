using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               bart_options(num_trees=100, num_draws=1000, small_leaf_likelihood_factor=0.001),
               bart_options(num_trees=100, num_draws=1000, small_leaf_likelihood_factor=0.01 ),
               bart_options(num_trees=100, num_draws=1000, small_leaf_likelihood_factor=0.1  ),
               bart_options(num_trees=100, num_draws=1000, small_leaf_likelihood_factor=1.0  )]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, options, ["0.001", "0.01", "0.1", "1.0"], cor, iterations=100)

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      P90=quantile(df[:Score], 0.10),
                                      P75=quantile(df[:Score], 0.25),
                                      P50=quantile(df[:Score], 0.50),
                                      P25=quantile(df[:Score], 0.75),
                                      P10=quantile(df[:Score], 0.90)))
println(dist)
