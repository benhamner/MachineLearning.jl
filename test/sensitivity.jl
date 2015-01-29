using Base.Test
using DataFrames
using MachineLearning
using RDatasets

x = randn(200,3)
model = [10.0;1.0;0.0]
y = x*model

println(size(x))
println(size(y))

data = DataFrame(X1=vec(x[:,1]), X2=vec(x[:,2]), X3=vec(x[:,3]), Y=vec(y))

res = sensitivities(data, :Y, regression_forest_options())

@test res.data[1,:X1]>res.data[99,:X1]
