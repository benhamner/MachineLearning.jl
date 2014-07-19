using Gadfly
using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               bart_options(num_trees=10,   num_draws=1000),
               bart_options(num_trees=50,   num_draws=1000),
               bart_options(num_trees=100,  num_draws=1000),
               bart_options(num_trees=200,  num_draws=1000),
               bart_options(num_trees=500,  num_draws=1000),
               bart_options(num_trees=1000, num_draws=1000),]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, options, ["10 Trees", "50 Trees", "100 Trees", "200 Trees", "500 Trees", "1000 Trees"], cor)

draw(PNG("bart_num_trees.png", 8inch, 6inch), plot(res, x=:Name, y=:Score, Geom.boxplot))

dist = by(res, :Name, df -> DataFrame(Mean=mean(df[:Score]),
                                      Q1=quantile(df[:Score], 0.25),
                                      Q3=quantile(df[:Score], 0.75)))
println(dist)
