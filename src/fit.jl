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
    info = Vector{SplitInfo{T,Int}}(undef, 2^params.max_depth - 1)
    for leaf in 1:length(info)
        info[leaf] = SplitInfo{T,Int}(T(-Inf), zeros(T, 2 * K + 1), zeros(T, 2 * K + 1, X_size[2]), zeros(T, 2 * K + 1, X_size[2]), T(-Inf), T(-Inf), 0, 1, 0.0)
    end

    leaf_idx = ones(UInt16, X_size[1])

    cache = (params = deepcopy(params),
        X = X, Y = Y, pred = pred,
        𝑖_ = 𝑖_, 𝑗_ = 𝑗_, δ = δ,
        edges = edges, X_bin = X_bin,
        info = info,
        leaf_idx = leaf_idx,
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
        for leaf in 1:length(cache.info)
            cache.info[leaf].gainL = T(-Inf)
            cache.info[leaf].∑ .= 0
            cache.info[leaf].∑L .= 0
            cache.info[leaf].∑R .= 0
            cache.info[leaf].gainL = T(-Inf)
            cache.info[leaf].gainR = T(-Inf)
            cache.info[leaf].feat = 1
            cache.info[leaf].bin = 0
        end

        cache.leaf_idx .= 1

        # build a new tree
        update_grads!(params.loss, params.α, cache.pred, cache.Y, cache.δ)
        # initialize root
        @views cache.info[1].∑ .= reshape(sum(cache.δ[𝑖,:], dims=1), :)
        cache.info[1].gain = get_gain(params.loss, cache.info[1].∑, params.λ)
        # println("cache.info[1].∑: ", cache.info[1].∑)
        # println("cache.info[1].gain: ", cache.info[1].gain)

        tree = grow_tree(cache.δ, cache.hist, params, cache.info, cache.leaf_idx, cache.edges, cache.X_bin, 𝑖, 𝑗)
        push!(evotree.trees, tree)
        # predict!(cache.pred, tree, cache.X)

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
    leaf_idx,
    edges, X_bin, 𝑖, 𝑗) where {T <: AbstractFloat,U,S}

    K = 1
    depth = one(UInt16)
    tree = Tree(Vector{TreeNode{T,Int,Bool}}())

    # grow while there are remaining active nodes
    while depth <= params.max_depth
        # println("depth: ", depth)
        # grow nodes
        if depth == params.max_depth
            for leaf in 2^(depth - 1):2^(depth) - 1
                # add logic for whether a leaf should be added if growth had stopped at parent
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, info[leaf].∑, params, δ)))
            end
        else
            # println("id is left:", id)
            update_hist!(hist, δ, X_bin, 𝑖, 𝑗, K, leaf_idx)
            update_split_info!(info, hist, depth, 𝑗, K, params, edges)
            # with updated histogram - get the best split for every variable in every leaf in current depth
            for leaf in 2^(depth - 1):2^(depth) - 1
            #     # grow node if best split improves gain
            #     if info[leaf].bin > 0
            #         push!(tree.nodes, TreeNode(2 * leaf, 2 * leaf + 1, best.feat, best.cond, info.gain))
            #     else
                push!(tree.nodes, TreeNode(pred_leaf(params.loss, info[leaf].∑, params, δ)))
            #     end # end of single node split search
            end
            update_leaf_idx!(leaf_idx, 𝑖, info, X_bin)
            # println("size(𝑖):", size(𝑖))
        end
        depth += 1
    end # end of tree growth
    # println("info[1].feat: ", info[1].feat)
    # println("info[1].bin: ", info[1].bin)
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
