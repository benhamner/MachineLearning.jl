type PipelineOptions <: SupervisedModelOptions
    transformer_options::Vector{TransformerOptions}
    model_options::SupervisedModelOptions
end

type Pipeline <: SupervisedModel
    transformers::Vector{Transformer}
    model::SupervisedModel
end

function fit(x::Matrix{Float64}, y::Vector, opts::PipelineOptions)
    transformers = Array(transformer_options, 0)
    for opts=transformer_options
        push!(transformers, fit(x, opts))
    end
    # execute transformation
    # train model
    Pipeline(transformers, model)
end