using Base.Test
using MachineLearning

@test mean_squared_error([1.0,2.0],[1.0,3.0])==0.5
@test mean_squared_error([1.0,2.0,3.0],[1.0,2.0,3.0])==0.0
@test mean_squared_error([1.0,2.0,3.0],[1.0,2.0,4.0])==1.0/3
@test mean_squared_error([1.0,2.0],[1.0,6.0])==8.0

@test accuracy([1,2],[1,6])==0.5
@test accuracy(["Cat","Dog","Lion","Tiger"],["Cat","Fish","Lion","Tiger"])==0.75
@test accuracy([[1 2],[3 4]],[[1 2],[3 4]])==1.0
@test accuracy([[1 2],[3 4],[5 6]], [["a" "b"], ["c" "d"], ["e" "f"]])==0.0