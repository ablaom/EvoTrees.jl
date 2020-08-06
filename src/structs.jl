# store perf info of each variable
mutable struct SplitInfo{T<:AbstractFloat, S<:Int}
    gain::T
    ∑::Vector{T}
    ∑L::Matrix{T}
    ∑R::Matrix{T}
    gainL::T
    gainR::T
    bin::UInt8
    feat::S
    cond::T
end

function SplitInfo(T, J, K)
    SplitInfo{T,Int}(T(-Inf), zeros(T, 2 * K + 1), zeros(T, 2 * K + 1, J), zeros(T, 2 * K + 1, J), T(-Inf), T(-Inf), 0, 1, 0.0)
end

struct TrainNode{T<:AbstractFloat, S<:Int}
    parent::S
    depth::S
    ∑::Vector{T}
    gain::T
    𝑖::Vector{S}
    𝑗::Vector{S}
end

struct TreeNode{T<:AbstractFloat, S<:Int, B<:Bool}
    left::S
    right::S
    feat::S
    cond::T
    gain::T
    pred::Vector{T}
    split::B
end

TreeNode(left::S, right::S, feat::S, cond::T, gain::T) where {T<:AbstractFloat, S<:Int} = TreeNode{T,S,Bool}(left, right, feat, cond, gain, zeros(T,1), true)
TreeNode(pred::Vector{T}) where T = TreeNode(0, 0, 0, zero(T), zero(T), pred, false)
TreeNode(pred::T) where T = TreeNode(0, 0, 0, zero(T), zero(T), [pred], false)

# single tree is made of a root node that containes nested nodes and leafs
struct Tree{T<:AbstractFloat, S<:Int}
    nodes::Vector{TreeNode{T,S,Bool}}
end

# eval metric tracking
mutable struct Metric
    iter::Int
    metric::Float32
end
Metric() = Metric(0, Inf)

# gradient-boosted tree is formed by a vector of trees
struct GBTree{T<:AbstractFloat, S<:Int}
    trees::Vector{Tree{T,S}}
    params::EvoTypes
    metric::Metric
    K::Int
    levels
end
