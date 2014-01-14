type NeuralNetOptions
    bias_unit::Bool # include a bias unit that always outputs a +1
    hidden_layers::Vector{Int} # sizes of hidden layers
    num_passes::Int
end

neural_net_options(;bias_unit::Bool=true,
                   hidden_layers::Vector{Int}=[100],
                   num_passes::Int=100) = NeuralNetOptions(bias_unit, hidden_layers, num_passes)

type NeuralNetLayer
    weights::Array{Float64, 2}
end

type NeuralNet
    options::NeuralNetOptions
    layers::Vector{NeuralNetLayer}
    classes::Vector
end

sigmoid(z::Vector{Float64}) = 1/(1+exp(-z))
sigmoid_gradient(z::Vector{Float64}) = sigmoid(z) .* (1-sigmoid(z))

function train(x::Array{Float64, 2}, y::Vector, opts::NeuralNetOptions)
    num_features = size(x, 2)
    num_samples = size(x, 1)
    update_size = 1 / num_samples
    classes = sort(unique(y))
    classes_map = Dict(classes, [1:length(classes)])
    num_classes = length(classes)
    net = initialize_net(opts, classes, num_features)
    for i=1:opts.num_passes
        for j=1:num_samples
            actual = zeros(num_classes)
            actual[classes_map[y[j]]] = 1.0
            update_weights!(net, vec(x[j,:]), actual, update_size)
        end
    end
    net
end

function predict_probs(net::NeuralNet, sample::Vector{Float64})
    state = sample
    for layer = net.layers
        if net.options.bias_unit==true
            state = [1.0;state]
        end

        state = sigmoid(layer.weights*state)
    end
    state
end

function predict(net::NeuralNet, sample::Vector{Float64})
    probs = predict_probs(net, sample)
    net.classes[minimum(find(x->x==maximum(probs), probs))]
end

function predict(net::NeuralNet, samples::Array{Float64, 2})
    [predict(net, vec(samples[i,:])) for i=1:size(samples,1)]
end

function update_weights!(net::NeuralNet, sample::Vector{Float64}, actual::Vector{Float64}, update_size::Float64)
    outputs = Array(Vector{Float64}, 0) # before passing through sigmoid
    activations = Array(Vector{Float64}, 0)
    push!(outputs, sample)
    push!(activations, sigmoid(sample))
    state = sample
    for layer = net.layers
        if net.options.bias_unit
            state = [1.0;state]
        end

        push!(outputs, layer.weights*state)
        state = sigmoid(outputs[length(outputs)])
        push!(activations, state)
    end

    deltas = activations[length(activations)] - actual
    for i=length(net.layers):-1:1
        gradient = update_size*deltas*(net.options.bias_unit?hcat(1,activations[i]'):activations[i]')
        if i>1
            deltas = net.layers[i].weights'*deltas
            if net.options.bias_unit
            	deltas = deltas[2:length(deltas)]
            end
            deltas = deltas.*sigmoid_gradient(outputs[i])
        end
        net.layers[i].weights -= gradient
    end
end

function initialize_layer(number_in::Int, number_out::Int)
    epsilon_init = sqrt(6) / sqrt(number_in + number_out)
    weights = 2.0 * (rand(number_out, number_in) - 0.5) * epsilon_init
    NeuralNetLayer(weights)
end

function initialize_net(opts::NeuralNetOptions, classes::Vector, num_features::Int)
    layers = Array(NeuralNetLayer, 0)
    if isempty(opts.hidden_layers)
        push!(layers, initialize_layer(num_features + (opts.bias_unit?1:0), length(classes)))
    else
        push!(layers, initialize_layer(num_features + (opts.bias_unit?1:0), opts.hidden_layers[1]))
        for i=1:length(opts.hidden_layers)-1
            push!(layers, initialize_layer(opts.hidden_layers[i] + (opts.bias_unit?1:0), opts.hidden_layers[i+1]))
        end
        push!(layers, initialize_layer(opts.hidden_layers[length(opts.hidden_layers)] + (opts.bias_unit?1:0), length(classes)))
    end
    NeuralNet(opts, layers, classes)
end