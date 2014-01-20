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
    RandomForest,
    RandomForestOptions,

    # methods
    accuracy,
    cost,
    cost_gradient!,
    decision_tree_options,
    depth,
    gini,
    initialize_net,
    log_loss,
    mean_log_loss,
    mean_squared_error,
    net_to_weights,
    neural_net_options,
    one_hot,
    predict,
    predict_probs,
    random_forest_options,
    split_location,
    split_train_test,
    train,
    weights_to_net!

include("decision_tree.jl")
include("metrics.jl")
include("neural_net.jl")
include("random_forest.jl")
include("sample.jl")

end