using Base.Test
using DataFrames
using MachineLearning
using RDatasets

x = randn(200,3)
model = [10.0;1.0;0.0]
y = x*model

data = DataFrame(X1=vec(x[:,1]), X2=vec(x[:,2]), X3=vec(x[:,3]), Y=vec(y))

res = sensitivities(data, :Y, regression_forest_options())

println("X1: ", mean(res.data[:X1]))
println("X2: ", mean(res.data[:X2]))
println("X3: ", mean(res.data[:X3]))

@test mean(res.data[:X1])>mean(res.data[:X2])
@test mean(res.data[:X2])>mean(res.data[:X3])
