tests = [
    "common",
    "decision_tree",
    "metrics",
    "neural_net",
    "split",
    "random_forest"]

println("Running tests:")
cd("test")

for t in tests
    test_fn = "$t.jl"
    println(" * $test_fn")
    require(test_fn)
end