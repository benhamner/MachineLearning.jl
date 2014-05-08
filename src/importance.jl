type ImportanceResults
    names::Vector{String}
    importances::Vector{Float64}
    best_score::Float64
end

function importances(x::Matrix{Float64}, y::Vector, opts::ClassificationModelOptions)
    num_features = size(x, 2)
    num_splits   = 5
    imps = zeros(num_splits, num_features)
    bests = zeros(num_splits)
    for i=1:num_splits
        x_train, y_train, x_test, y_test = split_train_test(x, y, seed=i)
        imp, best = single_importances(x_train, y_train, x_test, y_test, opts)
        imps[i,:] = imp
        bests[i]  = best
    end
    names = [@sprintf("X%d", i) for i=1:num_features]
    ImportanceResults(names, vec(mean(imps, 1)), mean(bests))
end

function single_importances(x_train::Matrix{Float64}, y_train::Vector, x_test::Matrix{Float64}, y_test::Vector, opts::ClassificationModelOptions)
    num_features = size(x_train, 2)
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
    importance, best_score
end

function importances(df::DataFrame, target_column::Symbol, opts::ClassificationModelOptions)
    y = array(df[target_column])
    if typeof(opts) <: RegressionModelOptions
        y *= 1.0
    end
    columns = filter(x->x!=target_column, names(df))
    x = float_matrix(df[columns])
    results = importances(x, y, opts)
    results.names = [string(c) for c=columns]
    results
end

function Gadfly.plot(results::ImportanceResults)
    importance = DataFrame(Feature=results.names, Importance=results.importances, Useful=[imp>0.0 ? "Yes":"No" for imp=results.importances])
    importance = sort(importance, cols=[:Feature])
    order = sortperm(importance[:Importance], rev=false)
    plot(importance,
         x="Importance",
         y="Feature",
         color="Useful",
         Geom.bar(orientation=:horizontal),
         Scale.y_discrete(order=order),
         Guide.title(@sprintf("Best Score: %0.4f", results.best_score)))
end
