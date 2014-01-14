function split_train_test(x::Array{Float64,2}, y::Vector, split_fraction::Float64=0.5)
    @assert size(x, 1)==length(y)
    @assert size(x, 1)>1
    @assert split_fraction>0.0
    @assert split_fraction<1.0

    i = shuffle([1:length(y)])
    cutoff = max(int(floor(split_fraction*length(y))), 1)
    x_train = x[i[1:cutoff], :]
    y_train = y[i[1:cutoff]]
    x_test  = x[i[cutoff+1:length(i)], :]
    y_test  = y[i[cutoff+1:length(i)]]
    x_train, y_train, x_test, y_test
end