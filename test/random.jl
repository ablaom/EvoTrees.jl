using DataFrames
using CSV
using Statistics
using StatsBase: sample
using Revise
using EvoTrees
using BenchmarkTools

# prepare a dataset
features = rand(100_000, 100)
# features = rand(100, 10)
X = features
Y = rand(size(X, 1))
𝑖 = collect(1:size(X,1))

# train-eval split
𝑖_sample = sample(𝑖, size(𝑖, 1), replace = false)
train_size = 0.8
𝑖_train = 𝑖_sample[1:floor(Int, train_size * size(𝑖, 1))]
𝑖_eval = 𝑖_sample[floor(Int, train_size * size(𝑖, 1))+1:end]

X_train, X_eval = X[𝑖_train, :], X[𝑖_eval, :]
Y_train, Y_eval = Y[𝑖_train], Y[𝑖_eval]

# train model
params1 = EvoTreeRegressor(
    loss=:logistic, metric=:logloss,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=1.0, colsample=1.0, nbins=32)

# linear: 1.202 s (648779 allocations: 466.69 MiB)
# logistic: 1.234 s (649778 allocations: 468.40 MiB)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n = 2)
@btime model = grow_gbtree($X_train, $Y_train, $params1, X_eval = $X_eval, Y_eval = $Y_eval)
@time pred_train = predict(model, X_train)
# linear: 7.597 ms (29 allocations: 626.58 KiB)
# logistic: 9.653 ms (32 allocations: 626.64 KiB)
@btime pred_train = predict(model, X_train)
mean(abs.(pred_train .- Y_train))

# train model
params1 = EvoTreeRegressor(
    loss=:logistic, metric=:logloss,
    nrounds=10,
    λ = 0.0, γ=0.0, η=0.1,
    max_depth = 6, min_weight = 1.0,
    rowsample=1.0, colsample=1.0, nbins=64)
@time model = grow_gbtree(X_train, Y_train, params1, X_eval = X_eval, Y_eval = Y_eval, print_every_n=2)
@time pred_train = predict(model, X_train)
