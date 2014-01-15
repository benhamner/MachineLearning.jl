function mean_squared_error(a::Array{Float64}, p::Array{Float64})
    mean((a-p).^2)
end

function accuracy(a::Array, b::Array)
    @assert size(a)==size(b)
    mean(map(x->x[1]==x[2], zip(a, b)))
end