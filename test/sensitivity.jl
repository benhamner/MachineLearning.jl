using Base.Test
using DataFrames
using MachineLearning
using RDatasets

x = randn(1000,3)
model = [10.0;1.0;0.0]
y = x*model

data = DataFrame(X1=vec(x[:,1]), X2=vec(x[:,2]), X3=vec(x[:,3]), Y=vec(y))

res = sensitivities(data, :Y, regression_forest_options())

for fea=[:X1, :X2, :X3]
    println(fea, ": ", mean(res.data[fea]), "\tSlope: ", mean(res.data[fea])/res.feature_ranges[fea])
end

@test mean(res.data[:X1])>mean(res.data[:X2])
@test mean(res.data[:X2])>mean(res.data[:X3])

x1_slope = mean(res.data[:X1])/res.feature_ranges[:X1]

@test x1_slope<15.0
@test x1_slope>7.5

res = sensitivities(data, :Y, bart_options())

for fea=[:X1, :X2, :X3]
    println(fea, ": ", mean(res.data[fea]), "\tSlope: ", mean(res.data[fea])/res.feature_ranges[fea])
end

#@test mean(res.data[:X1])>mean(res.data[:X2])
#@test mean(res.data[:X2])>mean(res.data[:X3])

x1_slope = mean(res.data[:X1])/res.feature_ranges[:X1]

#@test x1_slope<15.0
#@test x1_slope>7.5
