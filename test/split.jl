using Base.Test
using MachineLearning
using RDatasets

x = randn(1000,10)
y = randn(1000)
x_map = Dict([x[i,:] for i=1:size(x,1)], y)
split = split_train_test(x, y)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)
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

@test evaluate(split, regression_tree_options(), mean_squared_error)>0.0

split = split_train_test(x, vec(int(x[:,1]+x[:,2].>0.0)))
acc = evaluate(split, classification_tree_options(), accuracy)
println("Evaluate Train/Test Split Accuracy: ", acc)
@test acc>0.6

split = split_cross_valid(x, vec(int(x[:,1]+x[:,2].>0.0)))
acc = evaluate(split, classification_tree_options(), accuracy)
println("Evaluate Cross Validation Split Accuracy: ", acc)
@test acc>0.6

split = split_train_test(x, y, split_fraction=0.75, seed=1)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)
@test size(x_train, 1)==750
@test size(x_test,  1)==250
@test length(y_train) ==750
@test length(y_test)  ==250

split = split_train_test(x, y, split_fraction=0.75, seed=1)
a_train, b_train = train_set(split)
a_test,  b_test  = test_set(split)
@test a_train==x_train
@test b_train==y_train
@test a_test==x_test
@test b_test==y_test

split = split_train_test(x, y, split_fraction=0.75, seed=2)
a_train, b_train = train_set(split)
a_test,  b_test  = test_set(split)
@test a_train!=x_train
@test b_train!=y_train
@test a_test!=x_test
@test b_test!=y_test

split = split_cross_valid(x, y)
@test length(split.groups)==length(y)
@test sort(unique(split.groups))==Int[1:10]

x = [[1.0 2.0],[3.0 4.0]]
y = [1, 2]
split = split_train_test(x, y, split_fraction=0.75)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)
@test size(x_train, 1)==1
split = split_train_test(x, y, split_fraction=0.50)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)
@test size(x_train, 1)==1
split = split_train_test(x, y, split_fraction=0.25)
x_train, y_train = train_set(split)
x_test,  y_test  = test_set(split)
@test size(x_train, 1)==1

iris = dataset("datasets", "iris")
train, test = split_train_test(iris, split_fraction=0.50, seed=1)
@test nrow(train)==75
@test nrow(test) ==75

train2, test2 = split_train_test(iris, split_fraction=0.50, seed=1)
@test train2==train
@test test2==test

train2, test2 = split_train_test(iris, split_fraction=0.50, seed=2)
@test train2!=train
@test test2!=test

train, test = split_train_test(iris, split_fraction=2.0/3)
@test nrow(train)==100
@test nrow(test) ==50

u_iris = unique(iris)
train, test = split_train_test(u_iris, split_fraction=2.0/3)
data_map = [hash(u_iris[i,:])=>i for i=1:nrow(u_iris)]
rows = vcat([data_map[hash(train[i,:])]::Int for i=1:nrow(train)], [data_map[hash(test[i,:])]::Int for i=1:nrow(test)])
@test rows != [1:nrow(u_iris)]
@test sort(rows)==[1:nrow(u_iris)]

x = randn(100,3)
y = randn(100)
s1 = split_cross_valid(x, y, seed=1)
s2 = split_cross_valid(x, y, seed=1)
s3 = split_cross_valid(x, y, seed=2)
@test s1.groups==s2.groups
@test s1.groups!=s3.groups
@test sum(s1.groups.==1)==10

x_train, y_train = train_set(s1, 1)
@test size(x_train, 1)==90
@test length(y_train)==90
x_test, y_test = test_set(s1, 1)
@test size(x_test, 1)==10
@test length(y_test)==10

split = split_cross_valid(x, y, num_folds=2)
@test sort(unique(split.groups))==Int[1, 2]
@test sum(split.groups.==1)==50

split = split_cross_valid(x, y, num_folds=100)
@test sort(unique(split.groups))==[1:100]

@test_throws ErrorException split_cross_valid(x, y, num_folds=101)
