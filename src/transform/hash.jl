# Zero mean, unit variance transformer
type HashVectorizerOptions <: TransformerOptions
    bits::Int
end
hash_vectorizer_options(;num_bits::Int=12) = HashVectorizerOptions(num_bits)

type HashVectorizer <: Transformer
    options::HashVectorizerOptions
end

StatsBase.fit(xs::Vector, opts::HashVectorizerOptions) = HashVectorizer(opts)

function transform(hash_vectorizer::HashVectorizer, xs::Vector)
    res = zeros(length(xs), 2^hash_vectorizer.options.bits)

    for i in 1:length(xs)
        for f in xs[i]
            res[i, hash(f)>>(64-hash_vectorizer.options.bits)] += 1.0
        end
    end
    res
end