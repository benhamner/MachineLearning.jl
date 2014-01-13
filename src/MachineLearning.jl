module MachineLearning

export
    # types
    NeuralNet,
    NeuralNetLayer,
    NeuralNetOptions,

    # methods
    train,
    predict

include("neural_net.jl")

end