function compare(data_generator::Function, opts::Vector{SupervisedModelOptions}, opts_names::Vector, metric::Function; iterations::Int=10)
    res = DataFrame(Options=[], Iteration=[], Score=[])
    for i=1:iterations
        split = data_generator(i)
        for j=1:length(opts)
            println("Running ", opts_names[j], " on split ", i)
            res = vcat(res, DataFrame(Name=opts_names[j], Option=opts[j], Iteration=i, Score=evaluate(split, opts[j], metric)))
        end
    end
    res
end

function compare(data_generator::Function, opts_generator::Function, opts_sweep::Vector, metric::Function; iterations::Int=10)
    opts = SupervisedModelOptions[opts_generator(s) for s=opts_sweep]
    compare(data_generator, opts, opts_sweep, metric, iterations=iterations)
end
