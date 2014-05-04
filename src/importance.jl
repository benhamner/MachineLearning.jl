type ImportanceResults
    importances::Vector{Float64}
    best_score::Float64
end

function importances(x::Matrix{Float64}, y::Vector, opts::ClassificationModelOptions)
    num_features = size(x, 2)
    x_train, y_train, x_test, y_test = split_train_test(x, y)
    model      = fit(x_train, y_train, opts)
    predictions = vec(predict_probs(model, x_test)[:,2])
    best_score = auc(y_test, predictions)
    importance = zeros(num_features)
    for feature=1:num_features
        x_test_permuted = copy(x_test)
        x_test_permuted[:,feature] = shuffle(x_test[:,feature])
        predictions = vec(predict_probs(model, x_test_permuted)[:,2])
        importance[feature] = best_score-auc(y_test, predictions)
    end
    ImportanceResults(importance, best_score)
end
