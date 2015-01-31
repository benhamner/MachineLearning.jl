type PartialFeatureResults
    name::String
    values::Vector{Float64}
    responses::Vector{Float64}
end

function partials(model::ClassificationModel, x::Matrix{Float64}, feature::Int)
    x_copy = copy(x)
    values = quantile(x[:,feature], [0.01:0.01:0.99])
    responses = zeros(length(values))
    for i=1:length(values)
        x_copy[:,feature] = values[i]
        responses[i] = mean(predict_probs(model, x_copy)[:,2])
    end
    PartialFeatureResults(@sprintf("Feature %d", feature), values, responses)
end

function partials(x::Matrix{Float64}, y::Vector, feature::Int, opts::ClassificationModelOptions)
    model = fit(x, y, opts)
    partials(model, x, feature)
end

function partials(model::DataFrameModel, data::DataFrame, feature::Symbol)
    data_copy = copy(data)
    values = quantile(data[feature], [0.01:0.01:0.99])
    responses = zeros(length(values))
    for i=1:length(values)
        data_copy[feature] = values[i]
        if typeof(model.model)<:ClassificationModel
            responses[i] = mean(predict_probs(model, data_copy)[:,2])
        elseif typeof(model.model)<:RegressionModel
            responses[i] = mean(predict(model, data_copy))
        end
    end
    PartialFeatureResults(string(feature), values, responses)
end

function partials(data::DataFrame, target::Symbol, feature::Symbol, opts::SupervisedModelOptions)
    model = fit(data, target, opts)
    partials(model, data, feature)
end

function Gadfly.plot(results::PartialFeatureResults)
    df = DataFrame(Feature=results.values, Value=results.responses)

    plot(df,
         x="Feature",
         y="Value",
         Geom.line,
         Guide.title(results.name),
         Theme(background_color=color("white")))
end
