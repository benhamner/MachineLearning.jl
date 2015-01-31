type ImportanceResults
    names::Vector{String}
    importances::Vector{Float64}
    best_score::Float64
    all_importances::Matrix{Float64}
    best_scores::Vector{Float64}
end

function importances(x::Matrix{Float64}, y::Vector, opts::SupervisedModelOptions)
    num_features = size(x, 2)
    num_splits   = 5
    imps = zeros(num_splits, num_features)
    bests = zeros(num_splits)
    for i=1:num_splits
        split = split_train_test(x, y, seed=i)
        imp, best = single_importances(split, opts)
        imps[i,:] = imp
        bests[i]  = best
    end
    names = [@sprintf("X%d", i) for i=1:num_features]
    ImportanceResults(names, vec(mean(imps, 1)), mean(bests), imps, bests)
end

function single_importances(split::TrainTestSplit, opts::ClassificationModelOptions)
    x_train, y_train = train_set_x_y(split)
    x_test,  y_test  = test_set_x_y(split)
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

function single_importances(split::TrainTestSplit, opts::RegressionModelOptions)
    x_train, y_train = train_set_x_y(split)
    x_test,  y_test  = test_set_x_y(split)
    num_features = size(x_train, 2)
    model      = fit(x_train, y_train, opts)
    predictions = predict(model, x_test)
    best_score = rmse(y_test, predictions)
    importance = zeros(num_features)
    for feature=1:num_features
        x_test_permuted = copy(x_test)
        x_test_permuted[:,feature] = shuffle(x_test[:,feature])
        predictions = predict(model, x_test_permuted)
        importance[feature] = rmse(y_test, predictions)-best_score
    end
    importance, best_score
end

function importances(df::DataFrame, target_column::Symbol, opts::SupervisedModelOptions)
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
         Guide.title(@sprintf("Best Score: %0.4f", results.best_score)),
         Theme(background_color=color("white")))
end
