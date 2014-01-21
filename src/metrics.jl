function log_loss(a::Vector{Float64}, p::Vector{Float64})
    probs = copy(p)
    probs[map(x->x<1e-10, probs)] = 1e-10
    -sum(a.*log(probs)+(1.0-a).*log(1.0-probs))
end

function mean_log_loss(a::Matrix{Float64}, p::Matrix{Float64})
    @assert size(a)==size(p)
    probs = copy(p)
    probs[map(x->x<1e-10, probs)] = 1e-10
    -sum(a.*log(probs)+(1.0-a).*log(1.0-probs))/size(a,1)
end

function mean_squared_error(a::Array{Float64}, p::Array{Float64})
    mean((a-p).^2)
end

function accuracy(a::Array, p::Array)
    @assert size(a)==size(p)
    mean(map(x->x[1]==x[2], zip(a, p)))
end