using Base.Test

tests = [
    "bart",
    "classification",
    "common",
    "decision_tree",
    "importance",
    "metrics",
    "neural_net",
    "pipeline",
    "regression",
    "sensitivity",
    "split",
    "random_forest",
    "transform/zmuv",
    "transform/hash",
    "tree"]

println("Running tests:")

for my_test in [join([t,".jl"]) for t=tests]
    try
        include(my_test)
        println("\t\033[1m\033[32mPASSED\033[0m: $(my_test)")
    catch e
        anyerrors = true
        println("\t\033[1m\033[31mFAILED\033[0m: $(my_test)")
        rethrow(e)
    end
end
