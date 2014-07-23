abstract StopCriteria

type StopAfterIteration <: StopCriteria
    max_iteration::Int
end
StopAfterIteration() = StopAfterIteration(100)

type LogisticRegressionOptions <: ClassificationModelOptions
    bias_unit::Bool # include a bias unit that always outputs a +1
    train_method::Symbol
    learning_rate::Float64
    stop_criteria::StopCriteria
    display::Bool
    track_cost::Bool
end

function logistic_regression_options(;bias_unit::Bool=true,
                                     train_method::Symbol=:sgd,
                                     learning_rate::Float64=10.0,
                                     stop_criteria::StopCriteria=StopAfterIteration(),
                                     display::Bool=false,
                                     track_cost=false)
    LogisticRegressionOptions(bias_unit, train_method, learning_rate, stop_criteria, display, track_cost)
end

type LogisticRegression <: LogisticRegressionOptions
    options::ClassificationNetOptions
    weights::Matrix{Float64}
    classes::Vector
end

function classes(model::LogisticRegression)
    model.classes
end

function sigmoid(z::Array{Float64})
    1./(1+exp(-z))
end

function sigmoid_gradient(z::Array{Float64})
    sz = sigmoid(z)
    sz .* (1-sz)
end

function one_hot(y::Vector, classes_map::Dict)
    values = zeros(length(y), length(classes_map))
    for i=1:length(y)
        values[i, classes_map[y[i]]] = 1.0
    end
    values
end

# TODO