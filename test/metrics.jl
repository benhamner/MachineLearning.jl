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

@test_approx_eq log_loss([1.0,0.0], [0.8,0.2]) -(log(0.8)+log(0.8))
@test_approx_eq log_loss([0.0,1.0], [0.8,0.2]) -(log(0.2)+log(0.2))
@test_approx_eq log_loss([0.0,1.0], [0.3,0.7]) -(log(0.7)+log(0.7))
@test_approx_eq log_loss([1.0,0.0], [0.3,0.7]) -(log(0.3)+log(0.3))
a = [[0.0 0.0 1.0],[1.0 0.0 0.0],[0.0 1.0 0.0],[1.0 0.0 0.0]]
p = [[0.1 0.1 0.8],[0.2 0.4 0.4],[0.1 0.7 0.2],[0.3 0.4 0.4]]
@test_approx_eq mean_log_loss(a,p) mean([log_loss(vec(a[i,:]), vec(p[i,:])) for i=1:size(a,1)])