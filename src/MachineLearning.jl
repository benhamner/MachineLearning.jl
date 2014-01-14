module MachineLearning

export
    # types
    NeuralNet,
    NeuralNetLayer,
    NeuralNetOptions,

    # methods
    neural_net_options,
    train,
    predict

include("neural_net.jl")

end