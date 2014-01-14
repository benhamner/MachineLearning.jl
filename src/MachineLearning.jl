module MachineLearning

export
    # types
    NeuralNet,
    NeuralNetLayer,
    NeuralNetOptions,

    # methods
    neural_net_options,
    predict,
    split_train_test,
    train

include("neural_net.jl")
include("sample.jl")

end