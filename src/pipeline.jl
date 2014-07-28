type ClassificationPipelineOptions <: ClassificationModelOptions
    transformer_options::Vector{TransformerOptions}
    model_options::ClassificationModelOptions
end


type ClassificationPipeline <: ClassificationModel
    transformers::Vector{Transformer}
    model::ClassificationModel
end

type RegressionPipelineOptions <: RegressionModelOptions
    transformer_options::Vector{TransformerOptions}
    model_options::RegressionModelOptions
end

type RegressionPipeline <: RegressionModel
    transformers::Vector{Transformer}
    model::RegressionModel
end

function fit_predict(x::Matrix{Float64}, opts::Vector{TransformerOptions})
    transformers = Array(Transformer, 0)
    x_transformed = x
    for transformer_opts = opts
        transformer   = fit(x_transformed, transformer_opts)
        x_transformed = transform(transformer, x_transformed)
        push!(transformers, transformer)
    end
    transformers, x_transformed
end

function StatsBase.fit(x::Matrix{Float64}, y::Vector, opts::ClassificationPipelineOptions)
    transformers, x_transformed = fit_predict(x, opts.transformer_options)
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

function StatsBase.fit(x::Matrix{Float64}, y::Vector, opts::RegressionPipelineOptions)
    transformers, x_transformed = fit_predict(x, opts.transformer_options)
    model = fit(x_transformed, y, opts.model_options)
    RegressionPipeline(transformers, model)
end

function StatsBase.predict(pipeline::RegressionPipeline, sample::Vector{Float64})
    for transformer = pipeline.transformers
        sample = transform(transformer, sample)
    end
    predict(pipeline.model, sample)
end
