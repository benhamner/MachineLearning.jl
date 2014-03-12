using DataFrames
using MachineLearning
using RDatasets

type RegressionAccuracy
    model_name::String
    julia_options::RegressionModelOptions
    r_script_name::ASCIIString
end

options = [bart_options(num_trees=10),
           regression_forest_options(num_trees=10)]

bart = RegressionAccuracy("BART", bart_options(num_trees=10), "bart.R")

datasets = [("car",      "Prestige",   :Prestige, 0.5),
            ("datasets", "quakes",     :Mag,      0.5)]

algorithms = [bart]

for algorithm=algorithms
    println("############ ", algorithm.model_name, " ############")
    for (pkg, dataset_name, colname, acc_threshold) = datasets
        println("- Dataset ", dataset_name)
        data = dataset(pkg, dataset_name)
        columns = filter(x->eltype(data[x]) <: Number, names(data))
        data = data[columns]

        train, test = split_train_test(data)
        ytest = [x for x=test[colname]]
        model = fit(train, colname, algorithm.julia_options)
        yhat = predict(model, test)
        acc = cor(ytest, yhat)
        println(@sprintf("Julia Correlation: %0.3f", acc), "\t", algorithm.julia_options)

        train = float_dataframe(train)
        test  = float_dataframe(test)

        train[:is_test] = "false"
        test[:is_test]  = "true"
        data_file    = tempname()
        results_file = tempname()

        writetable(data_file, rbind(train, test))

        run(Cmd(Union(UTF8String, ASCIIString)["Rscript", algorithm.r_script_name, results_file, data_file, string(colname)]))
        r_results, header = readcsv(results_file, Float64, has_header=true)
        acc = cor(ytest, vec(r_results))
        println(@sprintf("R Correlation: %0.3f", acc), " ", algorithm.r_script_name)
    end
end