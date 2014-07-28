using Gadfly
using MachineLearning
using RDatasets

opts_generator(stop_iterations::Int) = RegressionPipelineOptions(TransformerOptions[ZmuvOptions()], regression_net_options(stop_criteria=StopAfterIteration(stop_iterations)))
opts_sweep = [100:100:2000]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, opts_generator, opts_sweep, cor)
draw(PNG("regression_net_stop_iterations.png", 8inch, 6inch), plot(res, x=:Name, y=:Score))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
