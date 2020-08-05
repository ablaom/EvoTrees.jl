# initialise evotree
function init_evotree(params::EvoTypes{T,U,S},
    X::AbstractMatrix, Y::AbstractVector; verbosity=1) where {T,U,S}

    seed!(params.seed)

    K = 1
    levels = ""
    X = convert(Matrix{T}, X)
    if typeof(params.loss) == Logistic
        Y = T.(Y)
        μ = logit(mean(Y))
    elseif typeof(params.loss) == Poisson
        Y = T.(Y)
        μ = log(mean(Y))
    elseif typeof(params.loss) == Softmax
        if typeof(Y) <: AbstractCategoricalVector
            levels = CategoricalArray(CategoricalArrays.levels(Y))
            K = length(levels)
            μ = zeros(T, K)
            Y = MLJModelInterface.int.(Y)
        else
            levels = CategoricalArray(sort(unique(Y)))
            K = length(levels)
            μ = zeros(T, K)
            Y = UInt32.(Y)
        end
    elseif typeof(params.loss) == Gaussian
        K = 2
        Y = T.(Y)
        μ = [mean(Y), log(std(Y))]
    else
        Y = T.(Y)
        μ = mean(Y)
    end

    # initialize preds
    pred = zeros(T, size(X, 1), K)
    fill!(pred, μ)

    bias = Tree([TreeNode([μ])])
    evotree = GBTree([bias], params, Metric(), K, levels)

    X_size = size(X)
    𝑖_ = collect(1:X_size[1])
    𝑗_ = collect(1:X_size[2])

    # initialize gradients and weights
    δ = zeros(T, X_size[1], 2 * K + 1)
    δ[:, 2 * K + 1] .= T(1)
    
    # binarize data into quantiles
    edges = get_edges(X, params.nbins)
    X_bin = binarize(X, edges)

    # initializde histograms
    hist = zeros(T, params.nbins, 2 * K + 1, X_size[2], 2^(params.max_depth - 1) - 1)

    # initialize splits info
    splits = Vector{SplitInfo{evotree.K,T,Int64}}(undef, X_size[2])
    for feat in 𝑗_
        splits[feat] = SplitInfo{T,Int}(T(-Inf), zeros(T, 2 * K + 1, X_size[2]), zeros(T, 2 * K + 1, X_size[2]), T(-Inf), T(-Inf), 0, feat, 0.0)
    end

    cache = (params = deepcopy(params),
        X = X, Y = Y, pred = pred,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, δ = δ,
        edges = edges, X_bin = X_bin,
        info = info,
        hist = hist)

    cache.params.nrounds = 0

    return evotree, cache
end


function grow_evotree!(evotree::GBTree{T,S}, cache; verbosity=1) where {T,S}

    # initialize from cache
    params = evotree.params
    X_size = size(cache.X_bin)
    δnrounds = params.nrounds - cache.params.nrounds

    # loop over nrounds
    for i in 1:δnrounds

        # select random rows and cols
        𝑖 = cache.𝑖_[sample(cache.𝑖_, ceil(Int, params.rowsample * X_size[1]), replace=false, ordered=true)]
        𝑗 = cache.𝑗_[sample(cache.𝑗_, ceil(Int, params.colsample * X_size[2]), replace=false, ordered=true)]
        # reset gain to -Inf
        for feat in cache.𝑗_
            cache.info[feat].gain = T(-Inf)
        end

        # build a new tree
        update_grads!(params.loss, params.α, cache.pred, cache.Y, cache.δ, cache.δ², cache.𝑤)
        ∑ = reshape(sum(cache.δ[𝑖,:], dims=1), :)
        gain = get_gain(params.loss, ∑, params.λ)
        # assign a root and grow tree
        cache.info[1] = SplitInfo(0, 1, ∑, gain, 𝑖, 𝑗)
        # train_nodes[1] = TrainNode(0, 1, ∑, gain, 𝑖, 𝑗)
        tree = grow_tree(cache.δ, cache.hist, params, cache.info, cache.edges, cache.X_bin)
        push!(evotree.trees, tree)
        predict!(cache.pred, tree, cache.X)

    end # end of nrounds

    cache.params.nrounds = params.nrounds
    # cache = (deepcopy(params), X, Y, pred, 𝑖_, 𝑗_, δ, δ², 𝑤, edges, X_bin, train_nodes, splits, hist_δ, hist_δ², hist_𝑤)
    # return model, cache
    return evotree
end

# grow a single tree
function grow_tree(δ, hist,
    params::EvoTypes{T,U,S},
    info::Vector{SplitInfo{T,Int}},
    edges, X_bin) where {T <: AbstractFloat,U,S}

    K = 1
    depth = 1
    tree = Tree(Vector{TreeNode{T,Int,Bool}}())

    # grow while there are remaining active nodes
    while depth <= params.max_depth
        # grow nodes
        if depth == params.max_depth
            for leaf in 2^(depth - 1):2^(depth) - 1
                # add logic for whether a leaf should be added if growth had stopped at parent
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ²)))
            end
        else
            # println("id is left:", id)
            update_hist!(hist, δ, X_bin, 𝑖, 𝑗, params.K, leaf_idx)
            # with updated histogram - get the best split for every variable in every leaf in current depth
            for leaf in 2^(depth - 1):2^(depth) - 1
                update_split_info!(info, hist, depth, 𝑗, K, params, edges)

                # grow node if best split improves gain
                if best.gain > node.gain + params.γ
                    left, right = update_set(node.𝑖, best.𝑖, view(X_bin, :, best.feat))
                    # train_nodes[leaf_count + 1] = TrainNode(id, node.depth + 1, best.∑L, best.gainL, left, node.𝑗)
                    # train_nodes[leaf_count + 2] = TrainNode(id, node.depth + 1, best.∑R, best.gainR, right, node.𝑗)
                    push!(tree.nodes, TreeNode(leaf_count + 1, leaf_count + 2, best.feat, best.cond, best.gain - node.gain))
                else
                    push!(tree.nodes, TreeNode(pred_leaf(params.loss, node, params, δ)))
                end # end of single node split search
            
            end
            
            
        end
        depth += 1
    end # end of tree growth
    return tree
end

# extract the gain value from the vector of best splits and return the split info associated with best split
function get_max_gain(splits::Vector{SplitInfo{T,S}}) where {T,S}
    gains = (x -> x.gain).(splits)
    feat = findmax(gains)[2]
    best = splits[feat]
    return best
end

function fit_evotree(params, X_train, Y_train;
    X_eval=nothing, Y_eval=nothing,
    early_stopping_rounds=9999,
    eval_every_n=1,
    print_every_n=9999,
    verbosity=1)

    # initialize metric
    iter_since_best = 0
    if params.metric != :none
        metric_track = Metric()
        metric_best = Metric()
    end

    nrounds_max = params.nrounds
    params.nrounds = 0
    model, cache = init_evotree(params, X_train, Y_train)
    iter = 1

    if params.metric != :none && X_eval !== nothing
        pred_eval = predict(model.trees[1], X_eval, model.K)
        Y_eval = convert.(eltype(cache.Y), Y_eval)
    end

    while model.params.nrounds < nrounds_max && iter_since_best < early_stopping_rounds
        model.params.nrounds += 1
        grow_evotree!(model, cache)
        # callback function
        if params.metric != :none
            if X_eval !== nothing
                predict!(pred_eval, model.trees[model.params.nrounds + 1], X_eval)
                metric_track.metric = eval_metric(Val{params.metric}(), pred_eval, Y_eval, params.α)
            else
                metric_track.metric = eval_metric(Val{params.metric}(), cache.pred, cache.Y, params.α)
            end
            if metric_track.metric < metric_best.metric
                metric_best.metric = metric_track.metric
                metric_best.iter =  model.params.nrounds
                iter_since_best = 0
            else
                iter_since_best += 1
            end
            if mod(model.params.nrounds, print_every_n) == 0 && verbosity > 0
                display(string("iter:", model.params.nrounds, ", eval: ", metric_track.metric))
            end
        end # end of callback
    end
    if params.metric != :none
        model.metric.iter = metric_best.iter
        model.metric.metric = metric_best.metric
    end
    params.nrounds = nrounds_max
    return model
end
