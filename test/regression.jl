using Base.Test
using MachineLearning
using RDatasets

options = [bart_options(num_trees=10),
           regression_forest_options(num_trees=10)]

datasets = [("car",      "Prestige",   :Prestige, 0.5),
            ("datasets", "quakes",     :Mag,      0.5)]
            #("plyr",     "baseball",   :R,        0.5)]
            #("Ecdat",    "BudgetFood", :WFood,    0.5)]

for (pkg, dataset_name, colname, score_threshold) = datasets
    println("- Dataset ", dataset_name)
    data = dataset(pkg, dataset_name)
    split = split_train_test(data, colname)
    for opts = options
        score = evaluate(split, opts, cor)
        println(@sprintf("Correlation: %0.3f", score), "\t", opts)
        @test score>score_threshold
    end
end