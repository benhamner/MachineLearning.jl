using Base.Test
using MachineLearning
using RDatasets

iris = data("datasets", "iris")
colnames = names(iris)
m_iris = float_matrix(iris[filter(x->x!="Species", colnames)])
@test m_iris[1,1]==5.1
@test m_iris[150,4]==1.8

forest = fit(iris, "Species", random_forest_options(num_trees=10, display=true))
@test predict(forest, iris[1,:])[1]=="setosa"