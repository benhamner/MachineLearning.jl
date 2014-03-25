using DataFrames
using MachineLearning
using RDatasets

type RegressionAccuracy
    model_name::String
    julia_options::RegressionModelOptions
    r_script_name::ASCIIString
    python_script_name::ASCIIString
end

bart          = RegressionAccuracy("BART",          bart_options(num_trees=10, num_draws=1000), "regression.R", "")
random_forest = RegressionAccuracy("Random Forest", regression_forest_options(num_trees=100),   "regression.R", "regression.py")
decision_tree = RegressionAccuracy("Decision Tree", regression_tree_options(),                  "",             "regression.py")

datasets = [("car",      "Prestige",   :Prestige, 0.5),
            ("datasets", "quakes",     :Mag,      0.5)]
            # ("ggplot2",  "movies",     :Rating,   0.5),

algorithms = [decision_tree,
              random_forest,
              bart]

for algorithm=algorithms
    println("ALGORITHM: ", algorithm.model_name)
    for (pkg, dataset_name, colname, acc_threshold) = datasets
        println(" - Dataset ", dataset_name)
        data = dataset(pkg, dataset_name)
        columns = filter(x->eltype(data[x]) <: Number, names(data))
        data = data[columns]

        train, test = split_train_test(data)
        ytest = [x for x=test[colname]]
        t0 = time()
        model = fit(train, colname, algorithm.julia_options)
        yhat = predict(model, test)
        t1 = time()
        acc = cor(ytest, yhat)
        println(@sprintf("   - Julia Correlation:  %0.3f\tElapsed Time: %0.2f", acc, t1-t0), "\t", algorithm.julia_options)

        train = float_dataframe(train)
        test  = float_dataframe(test)

        train[:is_test] = "false"
        test[:is_test]  = "true"
        data_file    = tempname()
        results_file = tempname()

        writetable(data_file, rbind(train, test))

        if algorithm.r_script_name != ""
            t0 = time()
            run(Cmd(Union(UTF8String, ASCIIString)["Rscript", algorithm.r_script_name, results_file, data_file, string(colname), algorithm.model_name]))
            t1 = time()
            r_results, header = readcsv(results_file, Float64, has_header=true)
            acc = cor(ytest, vec(r_results))
            println(@sprintf("   - R Correlation:      %0.3f\tElapsed Time: %0.2f", acc, t1-t0), "\t", algorithm.r_script_name)
        end

        results_file = tempname()
        if algorithm.python_script_name != ""
            t0 = time()
            run(Cmd(Union(UTF8String, ASCIIString)["python", algorithm.python_script_name, results_file, data_file, string(colname), algorithm.model_name]))
            t1 = time()
            python_results, header = readcsv(results_file, Float64, has_header=true)
            acc = cor(ytest, vec(python_results))
            println(@sprintf("   - Python Correlation: %0.3f\tElapsed Time: %0.2f", acc, t1-t0), "\t", algorithm.python_script_name)
        end
    end
end