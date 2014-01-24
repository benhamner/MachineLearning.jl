module MachineLearning

export
    # types
    ClassificationModel,
    DecisionBranch,
    DecisionNode,
    DecisionLeaf,
    DecisionTree,
    DecisionTreeOptions,
    NeuralNet,
    NeuralNetLayer,
    NeuralNetOptions,
    StopAfterIteration,
    RandomForest,
    RandomForestOptions,
    RegressionModel,
    StopAfterValidationErrorStopsImproving,
    SupervisedModel,
    SupervisedModelOptions,
    Transformer,

    # methods
    accuracy,
    cost,
    cost_gradient!,
    decision_tree_options,
    depth,
    fit,
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
    weights_to_net!

include("decision_tree.jl")
include("metrics.jl")
include("neural_net.jl")
include("random_forest.jl")
include("sample.jl")

end