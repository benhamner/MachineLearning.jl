using MachineLearning
using RDatasets

uncertainty_ranges(samples::Matrix{Float64}) = [quantile(vec(samples[i,:]), 0.9)-quantile(vec(samples[i,:]), 0.1) for i=1:size(samples, 1)]

split = split_train_test(SupervisedDataFrame(dataset("car", "Prestige"), :Prestige))

rf   = fit(split, regression_forest_options(num_trees=100))
rf_samples = zeros(length(test_set_y(split)), length(rf.model.trees))
for i=1:length(rf.model.trees)
    rf_samples[:,i] = predict(DataFrameModel(rf.model.trees[i], rf.colnames), split)
end

println("RF   P10-P90 Mean: ", mean(uncertainty_ranges(rf_samples)))
println("RF   P10-P90 Std:  ", std(uncertainty_ranges(rf_samples)))

bart = fit(split, bart_options(num_trees=100))
bart_samples = sample(bart, split)

println("BART P10-P90 Mean: ", mean(uncertainty_ranges(bart_samples)))
println("BART P10-P90 Std:  ", std(uncertainty_ranges(bart_samples)))
