using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               bart_options(num_trees=200, num_draws=1000, transform_probabilities=BartTreeTransformationProbabilies(0.5, 0.4, 0.1)),
               bart_options(num_trees=200, num_draws=1000, transform_probabilities=BartTreeTransformationProbabilies(0.5, 0.5, 0.0))]

#data = dataset("car", "Prestige")
data = dataset("datasets", "quakes")

data_generator(seed) = split_train_test(data, :Mag, seed=seed)

res = compare(data_generator, options, ["Default", "No Swap"], cor, iterations=100)

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      P95=quantile(df[:Score], 0.05),
                                      P75=quantile(df[:Score], 0.25),
                                      P25=quantile(df[:Score], 0.75),
                                      P05=quantile(df[:Score], 0.95)))
println(dist)
