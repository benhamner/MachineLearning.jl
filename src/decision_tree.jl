abstract DecisionNode

type DecisionTreeOptions

end

type DecisionLeaf <: DecisionNode
    probs::Vector{Float64}
end

type DecisionBranch <: DecisionNode
    feature::Int
    value::Float64
    left::DecisionNode
    right::DecisionNode
end

type DecisionTree
    root::DecisionNode
    classes::Vector
    options::DecisionTreeOptions
end

function train(x::Array{Float64,2}, y::Vector, opts::DecisionTreeOptions)

end

function train_branch(x::Array{Float64,2}, y::Vector{Int})

end