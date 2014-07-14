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

data = SupervisedDataFrame(iris, :Species)
x = data_set_x(data)
y = data_set_y(data)

@test length(y)==150
@test size(x, 1)==150
@test size(x, 2)==4
@test y[1]=="setosa"
@test x[1,1]==5.1

data = SupervisedDataFrame(iris, :SepalLength)
x = data_set_x(data)
y = data_set_y(data)

@test size(x, 2)==3
@test y[1]==5.1
@test x[1,1]==3.5

columns = data_frame_feature_columns(data)
@test in(:SepalWidth, columns)
@test in(:PetalLength, columns)
@test in(:PetalWidth, columns)
@test !in(:SepalLength, columns)

x = randn(100, 3)
y = randn(100)

data = SupervisedMatrix(x, y)
@test data_set_x(data)==x
@test data_set_y(data)==y
@test (x, y)==data_set_x_y(data)
