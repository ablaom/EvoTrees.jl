#############################################
# Get the braking points
#############################################
function get_edges(X::AbstractMatrix{T}, nbins=250) where {T}
    edges = Vector{Vector{T}}(undef, size(X, 2))
    @threads for i in 1:size(X, 2)
        edges[i] = quantile(view(X, :, i), (1:nbins) / nbins)
        if length(edges[i]) == 0
            edges[i] = [minimum(view(X, :, i))]
        end
    end
    return edges
end

####################################################
# Transform X matrix into a UInt8 binarized matrix
####################################################
function binarize(X, edges)
    X_bin = zeros(UInt8, size(X))
    @threads for i in 1:size(X, 2)
        X_bin[:,i] = searchsortedlast.(Ref(edges[i][1:end - 1]), view(X, :, i)) .+ 1
    end
    X_bin
end

# split row ids into left and right based on best split condition
function update_leaf_idx!(leaf_idx::Vector{T}, set, best_feat, best_cond, x_bin, depth::T) where {T}
    @inbounds for i in set
        left_id = leaf_idx[i] + 2^depth
        right_id = left_id + 1
        x_bin[i, best_feat[leaf_idx[i]]] <= best_cond[leaf_vec[i]] ? leaf_idx[i] = left_id : leaf_idx[i] = right_id
    end
end

# build histogram across all leafs within a depth
function update_hist!(hist::AbstractArray, δ::AbstractMatrix, X_bin::AbstractMatrix, 𝑖, 𝑗, K, leaf::AbstractVector)
    @inbounds @threads for j in 𝑗
        @inbounds for i in 𝑖
            @inbounds for k in 1:(2 * K + 1)
                hist[X_bin[i,j], k, j, leaf[i]] += δ[i, k]
            end
        end
    end
end

# find best splits across all leafs of a given depth
function update_split_info!(info::Vector{SplitInfo{T,S}}, hist::AbstractArray, depth, 𝑗, K, params::EvoTypes, edges::Vector{Vector{T}}) where {T,S}

    @inbounds for leaf in 2^(depth - 1):(2^depth - 1)
        @inbounds for j in 𝑗
            @inbounds for bin in 1:nbins
                @inbounds for k in 1:(2 * K + 1)
                    info[leaf].∑L[k,j] += hist[bin, k, j, leaf]
                    info[leaf].∑R[k,j] -= hist[bin, k, j, leaf]
                end

                gainL, gainR = get_gain(params.loss, info[leaf].∑L[:,j], params.λ), get_gain(params.loss, info[leaf].∑L[:,j], params.λ)
                gain = gainL + gainR

                if gain - params.γ > info[leaf].gain && ∑L[2 * K + 1] >= params.min_weight + 1e-12 && ∑R[2 * K + 1] >= params.min_weight + 1e-12
                    info[leaf].gain = gain
                    info[leaf].gainL = gainL
                    info[leaf].gainR = gainR
                    info[leaf].∑L = ∑L
                    info[leaf].∑R = ∑R
                    info[leaf].feat = 𝑗
                    info[leaf].cond = edges[j][bin]
                    info[leaf].𝑖 = bin
                end # info update if gain
            end # loop on bins
        end # loop on vars
    end # loop on leafs
end