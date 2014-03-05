tests = [
    "bart",
    "classification",
    "common",
    "decision_tree",
    "metrics",
    "neural_net",
    "pipeline",
    "regression",
    "split",
    "random_forest",
    "transform/zmuv",
    "tree"]

println("Running tests:")
cd("test")

for t in tests
    test_fn = "$t.jl"
    println(" * $test_fn")
    require(test_fn)
end