using Base.Test
using MachineLearning
using RDatasets

options = [classification_forest_options(),
           classification_tree_options(),
           classification_net_options()]

datasets = [("datasets", "iris", :Species, 0.6)]

for (pkg, dataset_name, colname, score_threshold) = datasets
    println("- Dataset ", dataset_name)
    data = dataset(pkg, dataset_name)
    split = split_train_test(data, colname)
    for opts = options
        score = evaluate(split, opts, accuracy)
        println(@sprintf("Accuracy: %0.3f", score), "\t", opts)
        @test score>score_threshold
    end
end