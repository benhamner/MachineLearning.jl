using Base.Test
using MachineLearning

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