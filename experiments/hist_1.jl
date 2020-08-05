using Statistics: mean, std, var
using StatsBase: sample
using StaticArrays
using Base.Threads: @threads
using BenchmarkTools
using Revise
using EvoTrees

n_obs = Int(5e5)
n_vars = 50
n_bins = 32
𝑖 = collect(1:n_obs);
𝑗 = collect(1:n_vars);
δ = rand(n_obs);
δ² = rand(n_obs);

hist_δ = zeros(n_bins, n_vars);
hist_δ² = zeros(n_bins, n_vars);
X_bin = rand(UInt8.(1:n_bins), n_obs, n_vars);
𝑖_4 = sample(𝑖, Int(n_obs / 4), ordered=true);

function iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)
    # hist_δ .*= 0.0
    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
        end
    end
end

@time iter_1(X_bin, hist_δ, δ, 𝑖, 𝑗)

# takeaway : significant speedup from depth 3 if building all hit simultaneously

@btime iter_1($X_bin, $hist_δ, $δ, $𝑖, $𝑗)
@btime iter_1($X_bin, $hist_δ, $δ, $𝑖_4, $𝑗)

#############################
# Original
#############################
δo = rand(SVector{1,Float32}, n_obs);
δ²o = rand(SVector{1,Float32}, n_obs);
hist_δo = zeros(SVector{1,Float32}, n_bins, n_vars);
hist_δ²o = zeros(SVector{1,Float32}, n_bins, n_vars);
function iter_ori(X_bin, hist_δ::Matrix{SVector{L,T}}, hist_δ²::Matrix{SVector{L,T}}, δ::Vector{SVector{L,T}}, δ²::Vector{SVector{L,T}}, 𝑖, 𝑗)  where {L,T}
    hist_δ .*= 0.0
    hist_δ² .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
            hist_δ²[X_bin[i,j], j] += δ²[i]
        end
    end
end

@time iter_ori(X_bin, hist_δo, hist_δ²o, δo, δ²o, 𝑖, 𝑗)
@btime iter_ori($X_bin, $hist_δo, $hist_δ²o, $δo, δ²o, $𝑖_4, $𝑗)
@btime iter_ori($X_bin, $hist_δo, $hist_δ²o, $δo, δ²o, $𝑖, $𝑗)


# try adding all info on single array rather than seperate vectors
function iter_1B(X_bin, hist_δ, hist_δ², δ, δ², 𝑖, 𝑗)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            hist_δ[X_bin[i,j], j] += δ[i]
            hist_δ²[X_bin[i,j], j] += δ²[i]
        end
    end
end

@btime iter_1B($X_bin, $hist_δ, $hist_δ², $δ, $δ², $𝑖, $𝑗)

# try adding all info on single array rather than seperate vectors
δ2 = rand(2, n_obs);
hist_δ2 = zeros(n_bins, 2, n_vars);
function iter_2(X_bin, hist_δ2, δ2, 𝑖, 𝑗)
    # hist_δ .*= 0.0
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            # view(hist_δ2, X_bin[i,j], j, :) .+= view(δ2, i, :)
            @inbounds for k in 1:2
                hist_δ2[X_bin[i,j], k, j] += δ2[k, i]
                # @inbounds hist_δ2[X_bin[i,j], 1, j] += δ2[i, 1]
                # @inbounds hist_δ2[X_bin[i,j], 2, j] += δ2[i, 2]
            end
        end
    end
end
@time iter_2(X_bin, hist_δ2, δ2, 𝑖, 𝑗)
@btime iter_2($X_bin, $hist_δ2, $δ2, $𝑖, $𝑗)


# integrate a leaf id
δ2 = rand(n_obs, 2);
hist = zeros(n_bins, 2, n_vars, 15);
@time hist .= 0;
function iter_3(X_bin, hist, δ, 𝑖, 𝑗, leaf)
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            # hist[X_bin[i,j], 1, j, 1] += δ[i, 1]
            # hist[X_bin[i,j], 2, j, 1] += δ[i, 2]
            @inbounds for k in 1:2
                hist[X_bin[i,j], k, j, leaf[i]] += δ[i, k]
            end
        end
    end
end

leaf_idx = ones(UInt16, n_obs);
leaf_idx = sample(UInt16.(8:15), n_obs);
@btime 𝑖b = BitSet($𝑖);
@time iter_3(X_bin, hist, δ2, 𝑖, 𝑗, leaf_idx);
@btime iter_3($X_bin, $hist, $δ2, $𝑖, $𝑗, leaf_idx);
@btime iter_3($X_bin, $hist, $δ2, $𝑖b, $𝑗, $2);