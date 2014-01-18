module MachineLearning

export
    # types
    DecisionBranch,
    DecisionNode,
    DecisionLeaf,
    DecisionTree,
    DecisionTreeOptions,
    NeuralNet,
    NeuralNetLayer,
    NeuralNetOptions,
    StopAfterIteration,
    StopAfterValidationErrorStopsImproving,

    # methods
    accuracy,
    cost,
    cost_gradient!,
    gini,
    initialize_net,
    log_loss,
    mean_log_loss,
    mean_squared_error,
    net_to_weights,
    neural_net_options,
    one_hot,
    predict,
    split_location,
    split_train_test,
    train,
    weights_to_net!

include("decision_tree.jl")
include("metrics.jl")
include("neural_net.jl")
include("sample.jl")

end