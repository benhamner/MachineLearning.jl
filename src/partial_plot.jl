function partial_plot(x::Matrix{Float64}, y::Vector, feature::Int, opts::ClassificationModelOptions)
    model = fit(x, y, opts)
    x_copy = copy(x)
    xs = quantile(x[:,feature], [0.01:0.01:0.99])
    ys = zeros(length(xs))
    for i=1:length(xs)
        x_copy[:,feature] = xs[i]
        ys[i] = mean(predict_probs(model, x_copy)[:,2])
    end
    df = DataFrame(Feature=xs, Probability=ys)

    plot(df,
         x="Feature",
         y="Probability",
         Geom.line)
end

function partial_plot(df::DataFrame, target::Symbol, feature::Symbol, opts::ClassificationModelOptions)
    y = array(df[target])
    columns = filter(x->x!=target, names(df))
    x = float_matrix(df[columns])
    partial_plot(x, y, find(columns.==feature)[1], opts)
end
