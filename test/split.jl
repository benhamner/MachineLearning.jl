using Base.Test
using MachineLearning
using RDatasets

x = randn(1000,10)
y = randn(1000)
x_map = Dict([x[i,:] for i=1:size(x,1)], y)
x_train, y_train, x_test, y_test = split_train_test(x, y)
@test size(x_train, 1)==500
@test size(x_test,  1)==500
@test length(y_train) ==500
@test length(y_test)  ==500
for i=1:size(x_train, 1)
    @test x_map[x_train[i,:]]==y_train[i]
end
for i=1:size(x_test, 1)
    @test x_map[x_test[i,:]]==y_test[i]
end

x_train, y_train, x_test, y_test = split_train_test(x, y, 0.75)
@test size(x_train, 1)==750
@test size(x_test,  1)==250
@test length(y_train) ==750
@test length(y_test)  ==250

x = [[1.0 2.0],[3.0 4.0]]
y = [1, 2]
x_train, y_train, x_test, y_test = split_train_test(x, y, 0.75)
@test size(x_train, 1)==1
x_train, y_train, x_test, y_test = split_train_test(x, y, 0.50)
@test size(x_train, 1)==1
x_train, y_train, x_test, y_test = split_train_test(x, y, 0.25)
@test size(x_train, 1)==1

iris = dataset("datasets", "iris")
train, test = split_train_test(iris, 0.50)
@test nrow(train)==75
@test nrow(test) ==75
train, test = split_train_test(iris, 2.0/3)
@test nrow(train)==100
@test nrow(test) ==50

u_iris = unique(iris)
train, test = split_train_test(u_iris, 2.0/3)
data_map = Dict([u_iris[i,:] for i=1:nrow(u_iris)], 1:nrow(u_iris))
rows = vcat([data_map[train[i,:]]::Int for i=1:nrow(train)], [data_map[test[i,:]]::Int for i=1:nrow(test)])
@test rows != [1:nrow(u_iris)]
@test sort(rows)==[1:nrow(u_iris)]