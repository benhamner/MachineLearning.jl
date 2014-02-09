module MachineLearning
    using
        DataFrames,
        Devectorize,
        Optim,
        RDatasets,
        StatsBase

    export
        # types
        ClassificationModel,
        ClassificationPipeline,
        ClassificationPipelineAny,
        DecisionBranch,
        DecisionNode,
        DecisionLeaf,
        DecisionTree,
        DecisionTreeOptions,
        NeuralNet,
        NeuralNetLayer,
        NeuralNetOptions,
        PipelineOptions,
        PipelineOptionsAny,
        StopAfterIteration,
        RandomForest,
        RandomForestOptions,
        RegressionModel,
        StopAfterValidationErrorStopsImproving,
        SupervisedModel,
        SupervisedModelOptions,
        Transformer,
        TransformerOptions,
        Zmuv,
        ZmuvOptions,

        # methods
        accuracy,
        cost,
        cost_gradient!,
        cost_gradient_update_net!,
        decision_tree_options,
        depth,
        fit,
        float_matrix,
        gini,
        initialize_net,
        initialize_neural_net_temporary,
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
        transform,
        weights_to_net!

    include("common.jl")
    include("decision_tree.jl")
    include("metrics.jl")
    include("neural_net.jl")
    include("pipeline.jl")
    include("random_forest.jl")
    include("split.jl")
    include("transform.jl")

end