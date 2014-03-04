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

depth{T}(tree::Tree{T}, node::Node{T}) = depth(tree.root, node) # depth of node in tree
function depth{T}(branch::Branch{T}, node::Node{T})
    if node==branch
        return 1
    end
    left_depth  = depth(branch.left,  node)
    right_depth = depth(branch.right, node)
    left_depth  = left_depth > 0  ? left_depth  + 1 : 0
    right_depth = right_depth > 0 ? right_depth + 1 : 0
    max(left_depth, right_depth)
end
depth{T}(leaf::Leaf{T}, node::Node{T}) = node==leaf ? 1 : 0

Base.length{T}(tree::Tree{T})     = length(tree.root)
Base.length{T}(branch::Branch{T}) = 1 + length(branch.left) + length(branch.right)
Base.length{T}(leaf::Leaf{T})     = 1

parent{T}(tree::Tree{T}, node::Node{T}) = parent(tree.root, node)
function parent{T}(branch::Branch{T}, node::Node{T})
    data_or_none(a, b) = a==None ? b : a
    if branch.left==node || branch.right==node
        this_parent = branch
    else
        this_parent = data_or_none(parent(branch.left, node), parent(branch.right, node))
    end
    this_parent
end
parent{T}(leaf::Leaf{T}, node::Node{T}) = None

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
leaves{T}(leaf::Leaf{T}) = Leaf{T}[leaf]

function branches{T}(branch::Branch{T})
    function branches!{T}(branch::Branch{T}, branch_nodes::Vector{Branch{T}})
        push!(branch_nodes, branch)
        branches!(branch.left,  branch_nodes)
        branches!(branch.right, branch_nodes)
    end
    branches!{T}(leaf::Leaf{T}, branch_nodes::Vector{Branch{T}}) = Nothing()

    branch_nodes = Branch{T}[branch]
    branches!(branch, branch_nodes)
    branch_nodes
end
branches{T}(tree::Tree{T}) = branches(tree.root)
branches{T}(leaf::Leaf{T}) = Leaf{T}[]