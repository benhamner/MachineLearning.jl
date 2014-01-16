module MachineLearning

export
    # types
    NeuralNet,
    NeuralNetLayer,
    NeuralNetOptions,
    StopAfterIteration,
    StopAfterValidationErrorStopsImproving,

    # methods
    accuracy,
    cost,
    cost_gradient!,
    initialize_net,
    log_loss,
    mean_log_loss,
    mean_squared_error,
    net_to_weights,
    neural_net_options,
    one_hot,
    predict,
    split_train_test,
    train,
    train_soph,
    weights_to_net!

include("metrics.jl")
include("neural_net.jl")
include("sample.jl")

end