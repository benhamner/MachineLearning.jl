benchmarks = [
    "neural_net"]

println("Running benchmarks:")
cd("benchmark")

for b in benchmarks
    benchmark_fn = "$b.jl"
    println(" * $benchmark_fn")
    require(benchmark_fn)
end