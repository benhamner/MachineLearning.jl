tests = [
    "accuracy",
    "bart",
    "common",
    "decision_tree",
    "metrics",
    "neural_net",
    "pipeline",
    "split",
    "random_forest",
    "transform/zmuv"]

println("Running tests:")
cd("test")

for t in tests
    test_fn = "$t.jl"
    println(" * $test_fn")
    require(test_fn)
end