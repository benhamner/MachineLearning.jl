abstract Tree{T}
abstract Node{T}
abstract Branch{T} <: Node{T}
abstract Leaf{T}   <: Node{T}

function valid_tree{T}(tree::Tree{T})
    @assert typeof(tree.root)<: Node
    @assert valid_node(tree.root)
    true
end

function valid_node{T}(branch::Branch{T})
    @assert valid_node(branch.left)
    @assert valid_node(branch.right)
    @assert typeof(branch.feature) <: Int
    @assert typeof(branch.value)   <: Float64
    true
end

function valid_node{T}(leaf::Leaf{T})
    true
end

depth{T}(tree::Tree{T})     = depth(tree.root)
depth{T}(branch::Branch{T}) = 1 + max(depth(branch.left), depth(branch.right))
depth{T}(leaf::Leaf{T})     = 1

Base.length{T}(tree::Tree{T})     = length(tree.root)
Base.length{T}(branch::Branch{T}) = 1 + length(branch.left) + length(branch.right)
Base.length{T}(leaf::Leaf{T})     = 1

function leaves{T}(branch::Branch{T})
    function leaves!{T}(branch::Branch{T}, leaf_nodes::Vector{Leaf{T}})
        leaves!(branch.left,  leaf_nodes)
        leaves!(branch.right, leaf_nodes)
    end
    function leaves!{T}(leaf::Leaf{T}, leaf_nodes::Vector{Leaf{T}})
        push!(leaf_nodes, leaf)
    end

    leaf_nodes = Array(Leaf{T}, 0)
    leaves!(branch, leaf_nodes)
    leaf_nodes
end
leaves{T}(tree::Tree{T}) = leaves(tree.root)
leaves{T}(leaf::Leaf{T}) = [leaf]