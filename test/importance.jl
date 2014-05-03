using Base.Test
using MachineLearning
using RDatasets

x = randn(1000,3)
model = [10.0;1.0;0.0]
y = Int[y>0.0?1:0 for y=x*model]

importance = importances(x, y, classification_forest_options())
@test importance[1]>importance[2]
@test importance[2]>importance[3]
@test importance[3]<0.01*importance[1]
