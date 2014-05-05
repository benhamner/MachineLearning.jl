using Base.Test
using MachineLearning
using RDatasets

x = randn(1000,3)
model = [10.0;1.0;0.0]
y = Int[y>0.0?1:0 for y=x*model]

res = importances(x, y, classification_forest_options())

@test res.importances[1]>res.importances[2]
@test res.importances[2]>res.importances[3]
@test res.importances[3]<0.01*res.importances[1]
