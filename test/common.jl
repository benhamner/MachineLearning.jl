using Base.Test
using MachineLearning
using RDatasets

iris = dataset("datasets", "iris")
colnames = names(iris)
m_iris = float_matrix(iris[filter(x->x!=:Species, colnames)])
@test m_iris[1,1]==5.1
@test m_iris[150,4]==1.8

split = split_train_test(iris, :Species)
forest = fit(train_set(split), classification_forest_options(num_trees=10))
yhat = predict(forest, test_set(split))
ytest = [x for x=(test_set(split).df[:Species])]
acc = accuracy(ytest, yhat)
@test acc>0.8