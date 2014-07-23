using Benchmark
using MachineLearning

require("../test/linear_data.jl")

num_features=5
x, y = linear_data(2500, num_features)
x_train, y_train, x_test, y_test = split_train_test(x, y)

opts = classification_net_options(hidden_layers=[20], learning_rate=10.0, stop_criteria=StopAfterIteration(5))
f() = fit(x_train, y_train, opts)

res = benchmark(f, "Neural Net", "Basic Train", 10)
for (colname, val)=res
    println(colname, "\t", val[1])
end

writetable("results/neural_net.csv", res)