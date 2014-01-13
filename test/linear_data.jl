function linear_data(num_samples::Int, num_features::Int)
    xs = randn(num_samples, num_features)
    model = randn(num_features)
    ys = int(map(x->x>0.0, xs*model))
    xs, ys
end