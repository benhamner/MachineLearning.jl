using Gadfly
using MachineLearning
using RDatasets

opts_generator(num_hidden::Int) = regression_net_options(hidden_layers=[num_hidden], stop_criteria=StopAfterIteration(1000), learning_rate=1.0)
opts_sweep = [10:10:100]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, opts_generator, opts_sweep, cor)
draw(PNG("regression_net_num_hidden.png", 8inch, 6inch), plot(res, x=:Name, y=:Score))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
