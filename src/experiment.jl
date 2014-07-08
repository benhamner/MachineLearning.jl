function compare(data_generator::Function, metric::Function, opts::Vector{SupervisedModelOptions}; iterations=10)
    res = DataFrame(Options=[], Iteration=[], Score=[])
    for i=1:iterations
        split = data_generator(i)
        for j=1:length(opts)
            res = vcat(res, DataFrame(Options=opts[j], Iteration=i, Score=evaluate(split, opts[j])))
        end
    end
    res
end
