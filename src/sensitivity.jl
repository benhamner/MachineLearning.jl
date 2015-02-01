type SensitivityResults
    data::DataFrame
    feature_ranges::Dict{Symbol, Float64}
end

function sensitivities(model::DataFrameModel, data::DataFrame, feature::Symbol)
    data_copy = copy(data)
    values = quantile(data[feature], [0.01:0.01:0.99])
    mins  = zeros(nrow(data))
    maxes = zeros(nrow(data))
    for i=1:length(values)
        data_copy[feature] = values[i]
        if typeof(model.model)<:ClassificationModel
            predictions = predict_probs(model, data_copy)[:,2]
        elseif typeof(model.model)<:RegressionModel
            predictions = predict(model, data_copy)
        end
        if i==1
            mins[:]  = predictions
            maxes[:] = predictions
        end
        mins[:]  = min(mins,  predictions)
        maxes[:] = max(maxes, predictions)
    end
    maxes-mins, maximum(values)-minimum(values)
end

function sensitivities(data::DataFrame, target::Symbol, opts::SupervisedModelOptions)
    model    = fit(data, target, opts)
    results  = DataFrame(Sample=[1:nrow(data)])
    features = filter(n->n!=target, names(data))
    feature_ranges = Dict{Symbol, Float64}()
    for feature=features
        results[feature], feature_ranges[feature] = sensitivities(model, data, feature)
    end
    SensitivityResults(results, feature_ranges)
end

function Gadfly.plot(results::SensitivityResults)
    data = melt(results.data, :Sample)
    rename!(data, :variable, :Feature)
    rename!(data, :value, :Sensitivity)
    sort!(data, cols=:Feature)
    order = sortperm(aggregate(data, :Feature, median)[:Sensitivity_median], rev=true)
    plot(data,
         x="Feature",
         y="Sensitivity",
         Geom.boxplot(),
         Scale.x_discrete(order=order),
         Theme(background_color=color("white")))
end
