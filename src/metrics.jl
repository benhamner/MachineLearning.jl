function log_loss(a::Vector{Float64}, p::Vector{Float64})
    -sum(a.*log(p)+(1.0-a).*log(1.0-p))
end

function mean_log_loss(a::Array{Float64,2}, p::Array{Float64,2})
    @assert size(a)==size(p)
    -sum(a.*log(p)+(1.0-a).*log(1.0-p))/size(a,1)
end

function mean_squared_error(a::Array{Float64}, p::Array{Float64})
    mean((a-p).^2)
end

function accuracy(a::Array, p::Array)
    @assert size(a)==size(p)
    mean(map(x->x[1]==x[2], zip(a, p)))
end