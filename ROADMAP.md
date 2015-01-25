Longer term Roadmap
===================

1. Dataframe is key underlying datatype (as opposed to Matrix{Float64})
2. Performance benchmarking and optimization
3. Robust sampling methods
4. Build out metrics, potentially break out as separate repo
5. Testing against more real world datasets

Short-term TODO's
=================

1. Partial/sensitivity refactor + unit tests
2. More unit tests
   - likelihood under different observations for bart.jl
   - split.jl
3. Graphical output for experiments
4. Experiments with multiple datasets
5. Support categorical predictors
6. Track & optionally show metadata and state during training 
7. Support minibatch sgd during neural net training
8. Supervised regression support for neural nets

Experiments To Run
==================

1. Impact of the jump distribution on time to convergence
2. Convergence of MCMC results
