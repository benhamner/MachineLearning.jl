using DataFrames
using MachineLearning
using RDatasets

type ClassificationAccuracy
    model_name::String
    julia_options::ClassificationModelOptions
    r_script_name::ASCIIString
    python_script_name::ASCIIString
end

random_forest = ClassificationAccuracy("Random Forest", classification_forest_options(num_trees=100),   "classification.R", "classification.py")
decision_tree = ClassificationAccuracy("Decision Tree", classification_tree_options(),                  "",                 "classification.py")
neural_net    = ClassificationAccuracy("Neural Net",    classification_net_options(),                           "",                 "")

datasets = [("ggplot2",  "midwest", :Category, 0.5),
            ("datasets", "iris",    :Species,  0.5)]

algorithms = [decision_tree,
              random_forest,
              neural_net]

for algorithm=algorithms
    println("ALGORITHM: ", algorithm.model_name)
    for (pkg, dataset_name, colname, acc_threshold) = datasets
        println(" - Dataset ", dataset_name)
        data = dataset(pkg, dataset_name)
        columns = filter(x->eltype(data[x]) <: Number || x == colname, names(data))
        data = data[columns]

        train, test = split_train_test(data)
        ytest = [x for x=test[colname]]
        t0 = time()
        model = fit(train, colname, algorithm.julia_options)
        yhat = predict(model, test)
        t1 = time()
        acc = accuracy(ytest, yhat)
        println(@sprintf("   - Julia Accuracy:  %0.3f\tElapsed Time: %0.2f", acc, t1-t0), "\t", algorithm.julia_options)

        #train = float_dataframe(train)
        #test  = float_dataframe(test)

        train[:is_test] = "false"
        test[:is_test]  = "true"
        data_file    = tempname()
        results_file = tempname()

        writetable(data_file, rbind(train, test))

        if algorithm.r_script_name != ""
            t0 = time()
            run(Cmd(Union(UTF8String, ASCIIString)["Rscript", algorithm.r_script_name, results_file, data_file, string(colname), algorithm.model_name]))
            t1 = time()
            r_results, header = readcsv(results_file, ASCIIString, has_header=true)
            r_results = [strip(x, '"') for x=vec(r_results)]
            acc = accuracy(ytest, vec(r_results))
            println(@sprintf("   - R Accuracy:      %0.3f\tElapsed Time: %0.2f", acc, t1-t0), "\t", algorithm.r_script_name)
        end

        results_file = tempname()
        if algorithm.python_script_name != ""
            t0 = time()
            run(Cmd(Union(UTF8String, ASCIIString)["python", algorithm.python_script_name, results_file, data_file, string(colname), algorithm.model_name]))
            t1 = time()
            python_results, header = readcsv(results_file, ASCIIString, has_header=true)
            acc = accuracy(ytest, vec(python_results))
            println(@sprintf("   - Python Accuracy: %0.3f\tElapsed Time: %0.2f", acc, t1-t0), "\t", algorithm.python_script_name)
        end
    end
end