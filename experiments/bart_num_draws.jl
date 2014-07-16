using Gadfly
using MachineLearning
using RDatasets

opts_generator(num_draws::Int) = bart_options(num_trees=10, num_draws=num_draws)
opts_sweep = [100:100:500]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, opts_generator, opts_sweep, cor)
draw(PNG("bart_num_draws.png", 8inch, 6inch), plot(res, x=:Name, y=:Score))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
