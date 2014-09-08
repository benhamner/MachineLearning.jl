using Base.Test
using MachineLearning
using RDatasets

options = SupervisedModelOptions[
               bart_options(num_trees=10,  num_draws=10),
               bart_options(num_trees=100, num_draws=100)]

data = dataset("car", "Prestige")
data_generator(seed) = split_train_test(data, :Prestige, seed=seed)

res = compare(data_generator, options, ["10 Trees", "100 Trees"], cor)

@test res[:Name][1]=="10 Trees"
@test res[:Name][2]=="100 Trees"
@test res[:Score][2] > res[:Score][1]