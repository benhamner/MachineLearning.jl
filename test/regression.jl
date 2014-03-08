using Base.Test
using MachineLearning
using RDatasets

options = [bart_options(),
           regression_forest_options()]

datasets = [("car",      "Prestige",   :Prestige, 0.5),
            ("datasets", "quakes",     :Mag,      0.5),
            ("plyr",     "baseball",   :R,        0.5)]
            #("Ecdat",    "BudgetFood", :WFood,    0.5)]

for (pkg, dataset_name, colname, acc_threshold) = datasets
    println("- Dataset ", dataset_name)
    data = dataset(pkg, dataset_name)
    train, test = split_train_test(data)
    ytest = [x for x=test[colname]]
    for opts = options
        model = fit(train, colname, opts)
        yhat = predict(model, test)
        acc = cor(ytest, yhat)
        println(@sprintf("Correlation: %0.3f", acc), "\t", opts)
        @test acc>acc_threshold
    end
end