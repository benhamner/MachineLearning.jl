function log_loss(a::Vector{Float64}, p::Vector{Float64})
    probs = copy(p)
    probs[map(x->x<1e-10, probs)] = 1e-10
    -sum(a.*log(probs)+(1.0.-a).*log(1.0.-probs))
end

function mean_log_loss(a::Matrix{Float64}, p::Matrix{Float64})
    @assert size(a)==size(p)
    probs = copy(p)
    probs[map(x->x<1e-10, probs)] = 1e-10
    -sum(a.*log(probs)+(1.0.-a).*log(1.0.-probs))/size(a,1)
end

mse(a::Array{Float64}, p::Array{Float64}) = mean((a-p).^2)
rmse(a::Array{Float64}, p::Array{Float64}) = sqrt(mse(a, p))

function accuracy(a::Array, p::Array)
    @assert size(a)==size(p)
    mean(map(x->x[1]==x[2], zip(a, p)))
end

function auc(a::Array, p::Array)
    r = sortrank(p)
    res = ((sum(r[a.==1]) - sum(a.==1)*(sum(a.==1)+1)/2) /
           ( sum(a.<1)*sum(a.==1)))
    res
end

function sortrank(x)
    I = sortperm(x)
    r = 0*x;
    cur_val = x[I[1]];
    last_pos = 1;

    for i=1:length(I)
        if cur_val != x[I[i]]
            r[I[last_pos:i-1]] = (last_pos+i-1)/2;
            last_pos = i;
            cur_val = x[I[i]];
        end
        if i==length(I)
            r[I[last_pos:i]] = (last_pos+i)/2;
        end
    end
    r
end