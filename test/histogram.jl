using DataFrames
using CSV
using Statistics
using Base.Threads: @threads
using StatsBase: sample
using StaticArrays
using Revise
using BenchmarkTools
using EvoTrees
using EvoTrees: get_gain, get_edges, binarize, get_max_gain, update_grads!, grow_tree, grow_gbtree, SplitInfo, Tree, TrainNode, TreeNode, EvoTreeRegressor, predict, predict!, sigmoid
using EvoTrees: find_bags, update_bags!
using EvoTrees: find_split_static!, pred_leaf

# prepare a dataset
features = rand(100_000, 100)
# features = rand(1_000 10)
# x = cat(ones(20), ones(80)*2, dims=1)
# features =  hcat(x, features)

X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X,1))
𝑗 = collect(1:size(X,2))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]
𝑖 = collect(1:size(X_train,1))

# set parameters
params1 = EvoTreeRegressor(
    loss=:linear, metric=:mae,
    nrounds=1, nbins=32,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0, α=0.5,
    rowsample=1.0, colsample=1.0, K=1)

# initial info
@time δ = zeros(SVector{2*params1.K, Float64}, size(X_train, 1))
𝑤 = zeros(SVector{1, Float64}, size(X_train, 1)) .+ 1
pred = zeros(SVector{params1.K,Float64}, size(X_train,1))
@time update_grads!(params1.loss, params1.α, pred, Y_train, δ, 𝑤)
# linear: 65.300 μs (1 allocation: 16 bytes)
# logistic: 953.000 μs (1 allocation: 16 bytes)
# poisson: 401.699 μs (1 allocation: 16 bytes)
# quantile: 59.300 μs (1 allocation: 16 bytes)
# gaussian: 278.200 μs (2 allocations: 64 bytes)
# @btime update_grads!(params1.loss, params1.α, pred, Y_train, δ, 𝑤)
∑δ, ∑𝑤 = sum(δ[𝑖]), sum(𝑤[𝑖])
@time gain = get_gain(params1.loss, ∑δ, ∑𝑤, params1.λ)
# @btime gain = get_gain($params1.loss, $∑δ, $∑δ², $∑𝑤, $params1.λ)

# initialize train_nodes
train_nodes = Vector{TrainNode{2*params1.K, Float64, BitSet, Array{Int64, 1}, Int}}(undef, 2^params1.max_depth-1)
for node in 1:2^params1.max_depth-1
    train_nodes[node] = TrainNode(0, SVector{2*params1.K, Float64}(fill(-Inf, 2*params1.K)), SVector{1, Float64}(fill(-Inf, 1)), -Inf, BitSet([0]), [0])
    # train_nodes[feat] = TrainNode(0, fill(-Inf, params1.K), fill(-Inf, params1.K), -Inf, -Inf, BitSet([0]), [0])
end

# initializde node splits info and tracks - colsample size (𝑗)
splits = Vector{SplitInfo{2*params1.K, Float64, Int}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    splits[feat] = SplitInfo{2*params1.K, Float64, Int}(gain, SVector{2*params1.K, Float64}(zeros(2*params1.K)), SVector{1, Float64}(zeros(1)), SVector{2*params1.K, Float64}(zeros(2*params1.K)), SVector{1, Float64}(zeros(1)), -Inf, -Inf, 0, feat, 0.0)
end

# binarize data and create bags
@time edges = get_edges(X_train, params1.nbins)
@time X_bin = binarize(X_train, edges)
@time bags = Vector{Vector{BitSet}}(undef, size(𝑗, 1))
function prep(X_bin, bags)
    @threads for feat in 1:size(𝑗, 1)
         bags[feat] = find_bags(X_bin[:,feat])
    end
    return bags
end
@time bags = prep(X_bin, bags)

# initialize histograms
feat=1
hist_δ = Vector{Vector{SVector{2*params1.K, Float64}}}(undef, size(𝑗, 1))
hist_𝑤 = Vector{Vector{SVector{1, Float64}}}(undef, size(𝑗, 1))
for feat in 1:size(𝑗, 1)
    hist_δ[feat] = zeros(SVector{2*params1.K, Float64}, length(bags[feat]))
    hist_𝑤[feat] = zeros(SVector{1, Float64}, length(bags[feat]))
end

# grow single tree
#  0.135954 seconds (717.54 k allocations: 15.219 MiB)
@time train_nodes[1] = TrainNode(1, SVector(∑δ), SVector(∑𝑤), gain, BitSet(𝑖), 𝑗)
@time tree = grow_tree(bags, δ, 𝑤, hist_δ, hist_𝑤, params1, train_nodes, splits, edges, X_bin)
@time pred_train = predict(tree, X_train, params1.K)
# 965.899 μs (18 allocations: 626.08 KiB)
@btime pred_train = predict($tree, $X_train, $params1.K)
@time pred_leaf_ = pred_leaf(params1.loss, train_nodes[1], params1, δ)
# 1.899 ns (0 allocations: 0 bytes)
@btime pred_leaf_ = pred_leaf($params1.loss, $train_nodes[1], $params1, $δ)
# @btime pred_train = predict($tree, $X_train, params1.K)

@time model = grow_gbtree(X_train, Y_train, params1, print_every_n = 1)
# @btime model = grow_gbtree($X_train, $Y_train, $params1, print_every_n = 1)
@time pred_train = predict(model, X_train)
@btime pred_train = predict($model, $X_train)

params1 = Params(:linear, 10, λ, γ, 0.1, 5, min_weight, rowsample, colsample, nbins)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 1, metric=:mae)
@btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval, print_every_n = 1, metric=:mae)

@time pred_train = predict(model, X_train)
sqrt(mean((pred_train .- Y_train) .^ 2))

#############################################
# Quantiles with turbo
#############################################

𝑖_set = BitSet(𝑖);
@time bags = prep(X_bin, bags);
# target: find_split_turbo! in 0.001 sec for 100_000 observations
feat = 1
typeof(bags[feat][1])
# initialise node, info and tracks
train_nodes[1] = TrainNode(1, ∑δ, ∑𝑤, gain, BitSet(𝑖), 𝑗)
splits[feat] = SplitInfo{2*params1.K, Float64, Int}(gain, SVector{2*params1.K, Float64}(zeros(2*params1.K)), SVector{1, Float64}(zeros(1)), ∑δ, ∑𝑤, -Inf, -Inf, 0, feat, 0.0)

# 492.199 μs (343 allocations: 6.83 KiB)
# 1.038 ms (343 allocations: 6.83 KiB) for 200_000
splits[feat]
@time find_split_static!(hist_δ[feat], hist_𝑤[feat], bags[feat], view(X_bin,:,feat), δ, 𝑤, ∑δ, ∑𝑤, params1, splits[feat], edges[feat], train_nodes[1].𝑖)
# @code_warntype find_split_static!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], bags[feat], view(X_bin,:,feat), δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1, splits[feat], edges[feat], train_nodes[1].𝑖)
@btime find_split_static!($hist_δ[feat], $hist_𝑤[feat], $bags[feat], $view(X_bin,:,feat), $δ, $𝑤, $∑δ, $∑𝑤, $params1, $splits[feat], $edges[feat], $train_nodes[1].𝑖)

feat = 2
typeof(bags[feat][1])
# initialise node, info and tracks
train_nodes[1] = TrainNode(1, ∑δ, ∑δ², ∑𝑤, gain, BitSet(𝑖), 𝑗)
splits[feat] = SplitInfo{Float64, Int}(gain, SVector{params1.K, Float64}(zeros(params1.K)), SVector{params1.K, Float64}(zeros(params1.K)), SVector{1, Float64}(zeros(1)), ∑δ, ∑δ², ∑𝑤, -Inf, -Inf, 0, feat, 0.0)
@time find_split_static!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], bags[feat], view(X_bin,:,feat), δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1, splits[feat], edges[feat], train_nodes[1].𝑖)
@btime find_split_static!(hist_δ[feat], hist_δ²[feat], hist_𝑤[feat], bags[feat], view(X_bin,:,feat), δ, δ², 𝑤, ∑δ, ∑δ², ∑𝑤, params1, splits[feat], edges[feat], train_nodes[1].𝑖)


length(union(train_nodes[1].bags[1][1:13]...))
length(union(train_nodes[1].bags[1][1:13]...))
length(new_bags[2])
length(new_bags[2][1])
length(bags[2][32])
typeof(bags)
@btime update_bags_intersect($new_bags, $bags, $union(train_nodes[1].bags[1][1:13]...))
length(new_bags[2])
length(new_bags[2][2])
length(bags[2][1])

# extract the best feat from bags, and join all the underlying bins up to split point
best_bag = bags[1]
bins_L = union(best_bag[1:4]...)

function set_1(x, y)
    intersect!(x, y)
    return x
end

x = rand(UInt32, 100_000)
y = rand(x, 1000)

x_set = BitSet(x);
y_set = BitSet(y);

@btime set_1(x, y)
@btime set_1(x_set, y)


x = rand([1,2,3,4,5,6,7,8,9,10, 11,12], 1000)
x = rand(1000)
x_edges = quantile(x, (0:10)/10)
x_edges = unique(x_edges)
x_edges = x_edges[2:(end-1)]

length(x_edges)

x_bin = searchsortedlast.(Ref(x_edges), x) .+ 1
using StatsBase
x_map = countmap(x_bin)

x = reshape(x, (1000, 1))
x_edges = get_edges(x)
unique(quantile(view(X, :,i), (0:nbins)/nbins))[2:(end-1)]
x_bin = searchsortedlast.(Ref(x_edges[1]), x[:,1]) .+ 1
x_map = countmap(x_bin)

edges = get_edges(X, 32)
X_bin = zeros(UInt8, size(X))
@btime binindices(X[:,1], edges[1])
@btime X_bin = binarize(X, edges)

using StatsBase
x_map = countmap(x_bin)

x_edges[1]
