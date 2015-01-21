using Base.Test
using MachineLearning
using RDatasets

x = randn(1000,10)
y = randn(1000)
x_map = Dict([zip([x[i,:] for i=1:size(x,1)], y)...]) # TODO: cleanup post julia-0.3 compat
split = split_train_test(x, y)
train_data = train_set(split)
test_data  = test_set(split)
@test size(data_set_x(train_data), 1)==500
@test size(data_set_y(test_data),  1)==500
@test length(train_data.y) ==500
@test length(test_data.y)  ==500
for i=1:size(train_data.x, 1)
    @test x_map[train_data.x[i,:]]==train_data.y[i]
end
for i=1:size(test_data.x, 1)
    @test x_map[test_data.x[i,:]]==test_data.y[i]
end

@test evaluate(split, regression_tree_options(), mse)>0.0

split = split_train_test(x, vec(int(x[:,1]+x[:,2].>0.0)))
acc = evaluate(split, classification_tree_options(), accuracy)
println("Evaluate Train/Test Split Accuracy: ", acc)
@test acc>0.6

split = split_cross_valid(x, vec(int(x[:,1]+x[:,2].>0.0)))
acc = evaluate(split, classification_tree_options(), accuracy)
println("Evaluate Cross Validation Split Accuracy: ", acc)
@test acc>0.6

split = split_train_test(x, y, split_fraction=0.75, seed=1)
train_data = train_set(split)
test_data  = test_set(split)
@test size(train_data.x, 1)==750
@test size(test_data.x,  1)==250
@test length(train_data.y) ==750
@test length(test_data.y)  ==250

split = split_train_test(x, y, split_fraction=0.75, seed=1)
train_data_2 = train_set(split)
test_data_2  = test_set(split)
@test train_data.x==train_data_2.x
@test train_data.y==train_data_2.y
@test test_data.x==test_data_2.x
@test test_data.y==test_data_2.y

split = split_train_test(x, y, split_fraction=0.75, seed=2)
train_data_2 = train_set(split)
test_data_2  = test_set(split)
@test train_data.x!=train_data_2.x
@test train_data.y!=train_data_2.y
@test test_data.x!=test_data_2.x
@test test_data.y!=test_data_2.y

split = split_cross_valid(x, y)
@test length(split.groups)==length(y)
@test sort(unique(split.groups))==Int[1:10]

x = [[1.0 2.0],[3.0 4.0]]
y = [1, 2]
split = split_train_test(x, y, split_fraction=0.75)
train_data = train_set(split)
@test size(train_data.x, 1)==1
split = split_train_test(x, y, split_fraction=0.50)
train_data = train_set(split)
@test size(train_data.x, 1)==1
split = split_train_test(x, y, split_fraction=0.25)
train_data = train_set(split)
@test size(train_data.x, 1)==1

iris = dataset("datasets", "iris")
split = split_train_test(iris, :Species, split_fraction=0.50, seed=1)
@test nrow(train_set(split).df) == 75
@test nrow(test_set(split).df)  == 75

split2 = split_train_test(iris, :Species, split_fraction=0.50, seed=1)
@test split2.train_indices == split.train_indices
@test split2.test_indices  == split.test_indices

split3 = split_train_test(iris, :Species, split_fraction=0.50, seed=2)
@test split3.train_indices != split.train_indices
@test split3.test_indices  != split.test_indices

split = split_train_test(iris, :Species, split_fraction=2.0/3)
@test nrow(train_set(split).df)==100
@test nrow(test_set(split).df) ==50

u_iris = unique(iris)
split = split_train_test(u_iris, :Species, split_fraction=2.0/3)
train = train_set(split).df
test  = test_set(split).df
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
