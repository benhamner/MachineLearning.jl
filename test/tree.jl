using Base.Test
using MachineLearning

abstract Bad
type BadTree   <: Tree{Bad} end
type BadBranch <: Tree{Bad} end
type BadLeaf   <: Tree{Bad} end

@test_throws valid_node(BadLeaf())
@test_throws valid_node(BadBranch())
@test_throws valid_tree(BadTree())

abstract Good

type GoodTree <: Tree{Good}
    root::Node{Good}
end

type GoodBranch <: Branch{Good}
    feature::Int
    value::Float64
    left::Node{Good}
    right::Node{Good}
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