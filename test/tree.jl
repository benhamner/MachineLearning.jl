using Base.Test
using MachineLearning

abstract Bad
type BadTree   <: Tree{Bad} end
type BadBranch <: Tree{Bad} end
type BadLeaf   <: Tree{Bad} end

@test_throws MethodError    valid_node(BadLeaf())
@test_throws MethodError    valid_node(BadBranch())
@test_throws ErrorException valid_tree(BadTree())

abstract Good
typealias GoodNode Node{Good}

type GoodTree <: Tree{Good}
    root::GoodNode
end

type GoodBranch <: Branch{Good}
    feature::Int
    value::Float64
    left::GoodNode
    right::GoodNode
end

type GoodLeaf <: Leaf{Good}
    value::Float64
end

leaf1 = GoodLeaf(1.0)
leaf2 = GoodLeaf(2.0)
leaf3 = GoodLeaf(0.0)
leaf4 = GoodLeaf(1.0)
leaf5 = GoodLeaf(11.3)

branch1 = GoodBranch(1, 0.1, leaf1, leaf2)
branch2 = GoodBranch(1, 0.2, branch1, leaf3)
branch3 = GoodBranch(1, 0.3, leaf4, leaf5)
branch4 = GoodBranch(1, 0.4, branch2, branch3)

tree = GoodTree(branch4)

@test valid_node(leaf1)
@test valid_node(branch1)
@test valid_tree(tree)

@test length(leaf1)==1
@test length(leaf2)==1
@test length(branch1)==3
@test length(branch2)==5
@test length(branch3)==3
@test length(branch4)==9
@test length(tree)==9

@test depth(leaf1)==1
@test depth(branch1)==2
@test depth(branch2)==3
@test depth(branch3)==2
@test depth(branch4)==4
@test depth(tree)==4

@test depth(tree, leaf1)==4
@test depth(tree, leaf4)==3
@test depth(tree, branch1)==3
@test depth(tree, branch2)==2
@test depth(tree, branch3)==2
@test depth(tree, branch4)==1

@test parent(tree, leaf1)==branch1
@test parent(branch1, leaf1)==branch1
@test parent(branch2, leaf1)==branch1
@test parent(branch3, leaf1)==None
@test parent(tree, branch1)==branch2
@test parent(branch4, branch1)==branch2

@test Set(Any[leaves(tree)...])   ==Set(Any[[leaf1,leaf2,leaf3,leaf4,leaf5]...])
@test Set(Any[leaves(branch2)...])==Set(Any[[leaf1,leaf2,leaf3]...])
@test leaves(leaf1)==Leaf{Good}[leaf1]

@test Set(Any[branches(tree)...])   ==Set(Any[[branch1,branch2,branch3,branch4]...])
@test Set(Any[branches(branch2)...])==Set(Any[[branch1,branch2]...])
@test branches(leaf1)==Leaf{Good}[]

@test Set(Any[grand_branches(tree)...])    ==Set(Any[[branch4,branch2]...])
@test Set(Any[not_grand_branches(tree)...])==Set(Any[[branch3,branch1]...])
@test grand_branches(leaf1)==Branch{Good}[]
@test not_grand_branches(leaf1)==Branch{Good}[]
