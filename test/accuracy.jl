using Base.Test
using MachineLearning
using RDatasets

options = [classification_forest_options(),
           classification_tree_options(),
           neural_net_options()]

datasets = [("datasets", "iris", :Species, 0.7)]

for (pkg, dataset_name, colname, acc_threshold) = datasets
    println("- Dataset ", dataset_name)
    data = dataset(pkg, dataset_name)
    train, test = split_train_test(data)
    ytest = [x for x=test[colname]]
    for opts = options
        model = fit(train, colname, opts)
        yhat = predict(model, test)
        acc = accuracy(ytest, yhat)
        println(@sprintf("Accuracy: %0.3f", acc), "\t", opts)
        @test acc>acc_threshold
    end
end