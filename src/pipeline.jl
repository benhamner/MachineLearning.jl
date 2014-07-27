type PipelineOptions <: SupervisedModelOptions
    transformer_options::Vector{TransformerOptions}
    model_options::SupervisedModelOptions
end

# I have to write this since Julia types aren't covariant
function PipelineOptionsAny(transformer_options::Vector, model_options::SupervisedModelOptions)
    transformer_opts = Array(TransformerOptions, 0)
    for opts = transformer_options
        push!(transformer_opts, opts)
    end
    PipelineOptions(transformer_opts, model_options)
end

type ClassificationPipeline <: ClassificationModel
    transformers::Vector{Transformer}
    model::ClassificationModel
end

function StatsBase.fit(x::Matrix{Float64}, y::Vector, opts::PipelineOptions)
    transformers = Array(Transformer, 0)
    x_transformed = x
    for transformer_opts = opts.transformer_options
        transformer   = fit(x_transformed, transformer_opts)
        x_transformed = transform(transformer, x_transformed)
        push!(transformers, transformer)
    end
    model = fit(x_transformed, y, opts.model_options)
    ClassificationPipeline(transformers, model)
end

function predict_probs(pipeline::ClassificationPipeline, sample::Vector{Float64})
    for transformer = pipeline.transformers
        sample = transform(transformer, sample)
    end
    predict_probs(pipeline.model, sample)
end

function StatsBase.predict(pipeline::ClassificationPipeline, sample::Vector{Float64})
    for transformer = pipeline.transformers
        sample = transform(transformer, sample)
    end
    predict(pipeline.model, sample)
end