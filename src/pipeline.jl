type PipelineOptions <: SupervisedModelOptions
    transformer_options::Vector{TransformerOptions}
    model_options::SupervisedModelOptions
end

type ClassificationPipeline <: ClassificationModel
    transformers::Vector{Transformer}
    model::ClassificationModel
end

function fit(x::Matrix{Float64}, y::Vector, opts::PipelineOptions)
    transformers = Array(opts.transformer_options, 0)
    x_transformed = x
    for transformer_opts = opts.transformer_options
        transformer   = fit(x_transformed, transformer_opts)
        x_transformed = transform(transformer, x_transformed)
        push!(transformers, transformer)
    end
    model = fit(opts.model_options, x_transformed)
    ClassificationPipeline(transformers, model)
end

function StatsBase.predict_probs(pipeline::ClassificationPipeline, sample::Vector{Float64})

end

function StatsBase.predict(pipeline::ClassificationPipeline, sample::Vector{Float64})

end