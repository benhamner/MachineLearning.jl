function compare(data_generator::Function, opts::Vector{SupervisedModelOptions}, opts_names::Vector{ASCIIString}, metric::Function; iterations=10)
    res = DataFrame(Options=[], Iteration=[], Score=[])
    for i=1:iterations
        split = data_generator(i)
        for j=1:length(opts)
            res = vcat(res, DataFrame(Name=opts_names[j], Option=opts[j], Iteration=i, Score=evaluate(split, opts[j], metric)))
        end
    end
    res
end
