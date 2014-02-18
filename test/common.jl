using Base.Test
using MachineLearning
using RDatasets

iris = data("datasets", "iris")
colnames = names(iris)
m_iris = float_matrix(iris[filter(x->x!="Species", colnames)])
@test m_iris[1,1]==5.1
@test m_iris[150,4]==1.8

train, test = split_train_test(iris)
forest = fit(train, "Species", classification_forest_options(num_trees=10))
yhat = predict(forest, test)
ytest = [x for x=test["Species"]]
acc = accuracy(ytest, yhat)
@test acc>0.8