function linear_data(num_samples::Int, num_features::Int, noise::Float64=0.2)
    xs = randn(num_samples, num_features)
    model = randn(num_features)
    ys = int(map(x->x>0.0, xs*model+noise*randn(num_samples)))
    xs, ys
end

