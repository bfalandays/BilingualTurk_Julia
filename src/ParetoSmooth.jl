
module ParetoSmooth

export psis_loo

using ..Common, MCMCDiagnosticTools, AxisKeys, NamedDims

abstract type AbstractCV end

struct Psis{
    R <: Real,
    AT <: AbstractArray{R, 3},
    VT <: AbstractVector{R}
}
    weights::AT
    pareto_k::VT
    ess::VT
    sup_ess::VT
    r_eff::VT
    tail_len::AbstractVector{Int}
    posterior_sample_size::Int
    data_size::Int
end

function _calc_mcse(weights, log_likelihood, pointwise_loo, r_eff)
    pointwise_gmpd = exp_inline.(pointwise_loo)
    pointwise_var = zeros(eltype(log_likelihood), size(log_likelihood, 1))
    @inbounds for k = axes(weights,3), j = axes(weights,2), i = axes(weights,1)
        pointwise_var[i] += (weights[i,j,k] * (exp_inline(log_likelihood[i,j,k]) - pointwise_gmpd[i]))^2
    end
    # If MCMC draws follow a log-normal distribution, then their log has this std. error:
    @. pointwise_var = log1p(pointwise_var / pointwise_gmpd^2)
    # (google "log-normal method of moments" for a proof)
    # apply MCMC correlation correction:
    return @. sqrt(pointwise_var / r_eff)
end

struct PsisLoo{
    RealType <: Real,
    ArrayType <: AbstractArray{RealType},
    VectorType <: AbstractVector{RealType},
} <: AbstractCV
    estimates::KeyedArray{RealType, 2, <:NamedDimsArray, <:Any}
    pointwise::KeyedArray{RealType, 2, <:NamedDimsArray, <:Any}
    psis_object::Psis{RealType, ArrayType, VectorType}
    gmpd::RealType
    mcse::RealType
end

function relative_eff(
    sample::AbstractArray{<:Real,3}; 
    source::Union{AbstractString, Symbol}="default",
    maxlag=typemax(Int),
    kwargs...,
)
    if lowercase(String(source)) ∉ ("mcmc", "default")
        # Avoid type instability by computing the return type of `ess`
        T = promote_type(eltype(sample), typeof(zero(eltype(sample)) / 1))
        res = similar(sample, T, (axes(sample, 3),))
        return fill!(res, 1)
    end
    ess_sample = PermutedDimsArray(sample, (2, 3, 1))
    return MCMCDiagnosticTools.ess(ess_sample; maxlag, kwargs..., kind=:basic, relative=true)
end


function loo_from_psis(log_likelihood::AbstractArray{<:Real, 3}, psis_object::Psis)
    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)


    # TODO: Add a way of using score functions other than ELPD
    # log_likelihood::ArrayType = similar(log_likelihood)
    # log_likelihood .= score(log_likelihood)

    
    weights = psis_object.weights
    ξ = psis_object.pareto_k
    r_eff = psis_object.r_eff

    T = eltype(log_likelihood)
    pointwise_loo = zeros(T, size(log_likelihood, 1))
    pointwise_naive = zeros(T, size(log_likelihood, 1))
    @inbounds for k = axes(weights,3), j = axes(weights,2), i = axes(weights,1)
        pointwise_loo[i] += weights[i,j,k] * exp_inline(log_likelihood[i,j,k])
        pointwise_naive[i] += exp_inline(log_likelihood[i,j,k]-log_count)
    end
    for i = eachindex(pointwise_loo, pointwise_naive)
        pointwise_loo[i] = log(pointwise_loo[i])
        pointwise_naive[i] = log(pointwise_naive[i])
    end
    @inbounds for i = eachindex(pointwise_loo)
        acc_loo = zero(eltype(pointwise_loo))
        acc_naive = zero(eltype(pointwise_naive))
        for j = axes(weights,2), k = axes(weights,3)
            acc_loo += weights[i,j,k] * exp_inline(log_likelihood[i,j,k])
            # I'm assuming we don't want to reuse the `exp_inline(log_likelihood[i,j,k])` from above via
            # acc_naive += exp_inline(log_likelihood[i,j,k]) * inv_mcmc_count
            # for numerical reasons
            acc_naive += exp_inline(log_likelihood[i,j,k] - log_count)
        end
        pointwise_loo[i] = log(acc_loo)
        pointwise_naive[i] = log(acc_naive)
    end
    pointwise_p_eff = pointwise_naive - pointwise_loo
    pointwise_mcse = _calc_mcse(weights, log_likelihood, pointwise_loo, r_eff)

    pointwise = KeyedArray(
        hcat(pointwise_loo, pointwise_naive, pointwise_p_eff, pointwise_mcse, ξ);
        data=1:length(pointwise_loo),
        statistic=[:cv_elpd, :naive_lpd, :p_eff, :mcse, :pareto_k],
    )

    table = _generate_loo_table(pointwise)
    
    gmpd = exp_inline.(table(column=:mean, statistic=:cv_elpd))

    mcse = sum(abs2, pointwise_mcse) |> sqrt
    return PsisLoo(table, pointwise, psis_object, gmpd, mcse)
end


function loo_from_psis(
    log_likelihood::AbstractMatrix{<:Real}, psis_object::Psis, args...;
    chain_index::AbstractVector=_assume_one_chain(log_likelihood), kwargs...
)
    chain_index = Int.(chain_index)
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return loo_from_psis(new_log_ratios, psis_object, args...; kwargs...)
end


function _generate_loo_table(pointwise::AbstractMatrix{<:Real})

    data_size = size(pointwise, :data)
    # create table with the right labels
    table = KeyedArray(
        similar(NamedDims.unname(pointwise), 3, 4);
        statistic=[:cv_elpd, :naive_lpd, :p_eff],
        column=[:total, :se_total, :mean, :se_mean],
    )

    # calculate the sample expectation for the total score
    to_sum = pointwise([:cv_elpd, :naive_lpd, :p_eff])
    avgs = similar(to_sum, size(to_sum, 2))
    @inbounds for j = axes(to_sum,2)
        avg = zero(eltype(to_sum))
        @simd for i = axes(to_sum,1)
            avg += to_sum[i,j]
        end
        avgs[j] = avg / data_size
    end
    avgs = reshape(avgs, 3)
    table(:, :mean) .= avgs

    # calculate the sample expectation for the average score
    table(:, :total) .= table(:, :mean) .* data_size

    # calculate the sample expectation for the standard error in the totals
    se_mean = std(to_sum; mean=avgs', dims=1) / sqrt(data_size)
    se_mean = reshape(se_mean, 3)
    table(:, :se_mean) .= se_mean

    # calculate the sample expectation for the standard error in averages
    table(:, :se_total) .= se_mean * data_size

    if table(:p_eff, :total) ≤ 0
        @warn "The calculated effective number of parameters is negative, which should " *
        "not be possible. PSIS has failed to approximate the target distribution."
    end

    return table
end

function psis_loo(log_likelihood::AbstractArray{<:Real, 3}, args...; kwargs...)
    psis_object = psis(-log_likelihood, args...; kwargs...)
    return loo_from_psis(log_likelihood, psis_object)
end


function psis_loo(
    log_likelihood::AbstractMatrix{<:Real},
    args...;
    chain_index::AbstractVector=_assume_one_chain(log_likelihood),
    kwargs...,
)
    chain_index = Int.(chain_index)
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return psis_loo(new_log_ratios, args...; kwargs...)
end

function psis(
    log_ratios::AbstractArray{T, 3};
    source::Union{AbstractString, Symbol}="default",
    r_eff::AbstractVector{T}=relative_eff(log_ratios; source=source),
    calc_ess::Bool = true, 
    skip_checks::Bool = false
) where T <: Real

    dims = size(log_ratios)
    data_size = dims[1]
    post_sample_size = dims[2] * dims[3]

    skip_checks || _check_input_validity_psis(log_ratios)

    source = lowercase(String(source))
    if source == "default"
        @info "No source provided for samples; variables are assumed to be from a Markov " *
        "Chain. If the samples are independent, specify this with keyword argument " *
        "`source=:other`."
    end

    if !skip_checks && size(r_eff, 1) ≠ data_size
        throw(ArgumentError("Size of `r_eff` does not equal the number of data points."))
    end

    # Reshape to matrix (easier to deal with)
    
    weights = similar(log_ratios)
    weights_mat = reshape(weights, data_size, post_sample_size)
    @. weights = exp(log_ratios - $maximum(log_ratios; dims=(2,3)))


    tail_length = similar(r_eff, Int)
    ξ = similar(r_eff)
    @inbounds @views Threads.@threads for i in eachindex(tail_length)
        tail_length[i] = _def_tail_length(post_sample_size, r_eff[i])
        ξ[i] = psis!(
            weights_mat[i, :], r_eff[i]; 
            tail_length=tail_length[i], log_weights = false
        )
    end
    norm_const = zeros(eltype(weights), size(weights, 1))
    @inbounds for k = axes(weights, 3), j = axes(weights, 2), i = axes(weights, 1)
        norm_const[i] += weights[i, j, k]
    end
    @. weights = weights / norm_const

    
    if calc_ess
        ess = psis_ess(weights_mat, r_eff)
        inf_ess = sup_ess(weights_mat, r_eff)
    else
        ess = similar(weights_mat, 0)
        inf_ess = similar(weights_mat, 0)
    end

    return Psis(
        weights,
        ξ, 
        ess, 
        inf_ess, 
        r_eff, 
        tail_length, 
        post_sample_size, 
        data_size
    )
end


function psis(
    log_ratios::AbstractMatrix{<:Real};
    chain_index::AbstractVector=_assume_one_chain(log_ratios),
    kwargs...,
)
    chain_index = Vector(Int.(chain_index))
    new_log_ratios = _convert_to_array(log_ratios, chain_index)
    return psis(new_log_ratios; kwargs...)
end


function psis(is_ratios::AbstractVector{<:Real}, args...; kwargs...)
    new_ratios = copy(is_ratios)
    ξ = psis!(new_ratios; kwargs...)
    return new_ratios, ξ
end

function psis!(is_ratios::AbstractVector{T}, r_eff::T=one(T);
    log_weights::Bool = true,
    tail_length::Integer = _def_tail_length(length(is_ratios), r_eff),
    skip_checks::Bool = false
) where T<:Real

    skip_checks || _check_input_validity_psis(is_ratios)
    
    len = length(is_ratios)
    tail_start = len - tail_length + 1  # index of smallest tail value

    # sort is_ratios and also get results of sortperm() at the same time
    ratio_index = collect(zip(is_ratios, Base.OneTo(len)))
    partialsort!(ratio_index, (tail_start-1):len; by=first)
    is_ratios .= first.(ratio_index)
    @views tail = is_ratios[tail_start:len]
    _check_tail(tail)
    cutoff = is_ratios[tail_start - 1]
    if log_weights 
        biggest = maximum(tail)
        @. is_ratios = exp(is_ratios - biggest)
        cutoff = exp(cutoff - biggest)
    end

    ξ = _psis_smooth_tail!(tail, cutoff, r_eff)

    # truncate at max of raw weights (1 after scaling)
    clamp!(is_ratios, 0, 1)
    # unsort the ratios to their original position:
    invpermute!(is_ratios, last.(ratio_index))

    if log_weights 
        @. is_ratios = log(is_ratios) + biggest
    end

    return ξ
end

function _def_tail_length(length::Integer, r_eff::Real=1)
    return min(cld(length, 5), ceil(3 * sqrt(length / r_eff))) |> Int
end

function _check_input_validity_psis(
    log_ratios::AbstractArray{<:Real}
)
    if any(_invalid_number, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains NaN  or inf values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `log_ratios` (array is empty)."))
    end
    return nothing
end

function _check_tail(tail::AbstractVector{T}) where {T <: Real}
    if tail[end] ≈ tail[1]
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution; all tail values are the " *
                "same. Likely causes are:\n$LIKELY_ERROR_CAUSES",
            ),
        )
    elseif length(tail) ≤ MIN_TAIL_LEN
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution; tail length was too " *
                "short. Likely causes are:\n$LIKELY_ERROR_CAUSES",
            ),
        )
    end
    return nothing
end

function _invalid_number(x::Real)
    return isinf(x) || isnan(x)
end

const MIN_TAIL_LEN = 5  # Minimum size of a tail for PSIS to give sensible answers

function _psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T, r_eff::T=one(T)) where {T <: Real}
    len = length(tail)
    if any(isinf.(tail))
        return ξ = Inf
    else
        @. tail = tail - cutoff

        # save time not sorting since tail is already sorted
        ξ, σ = gpd_fit(tail, r_eff)
        @. tail = gpd_quantile(($(1:len) - 0.5) / len, ξ, σ) + cutoff
    end
    return ξ
end

function gpd_fit(
    sample::AbstractVector{T},
    r_eff::T=1;
    wip::Bool=true,
    min_grid_pts::Integer=30,
    sort_sample::Bool=false,
) where T<:Real

    len = length(sample)
    # sample must be sorted, but we can skip if sample is already sorted
    if sort_sample
        sample = sort(sample; alg=QuickSort)
    end


    grid_size = min_grid_pts + isqrt(len)  # isqrt = floor sqrt
    n_0 = 10  # determines how strongly to nudge ξ towards .5
    x_star = inv(3 * sample[(len + 2) ÷ 4])  # magic number. ¯\_(ツ)_/¯
    invmax = inv(sample[len])

    # build pointwise estimates of ξ and θ at each grid point
    θ_hats = similar(sample, grid_size)
    @fastmath @. θ_hats = invmax + (1 - sqrt((grid_size + 1) / $(1:grid_size))) * x_star
    ξ_hats = similar(θ_hats)
    for i = eachindex(ξ_hats, θ_hats)
        ξ_hat = zero(eltype(ξ_hats))
        for j = eachindex(sample)
            ξ_hat += log1p(-θ_hats[i] * sample[j])
        end
        ξ_hats[i] = ξ_hat/len
    end

    log_like = ξ_hats  # Reuse preallocated array
    # Calculate profile log-likelihood at each estimate:
    for i = eachindex(ξ_hats, θ_hats)
        ξ_hats[i] = len * (log(-θ_hats[i] / ξ_hats[i]) - ξ_hats[i] - 1)
    end
    # Calculate weights from log-likelihood:
    weights = log_like  # Reuse preallocated array
    log_norm = StatsFuns.logsumexp(log_like)
    for i = eachindex(log_like)
        log_like[i] = exp_inline(log_like[i] - log_norm)
    end
    # Take weighted mean:
    θ_hat = zero(Base.promote_eltype(θ_hats, weights))
    @simd for i = eachindex(θ_hats, weights)
        θ_hat += θ_hats[i] * weights[i]
    end
    ξ = zero(θ_hat)
    @simd for i = eachindex(sample)
        ξ += log1p(-θ_hat * sample[i])
    end
    ξ /= len
    σ::T = -ξ / θ_hat

    # Drag towards .5 to reduce variance for small len
    if wip
        @fastmath ξ = (r_eff * ξ * len + 0.5 * n_0) / (r_eff * len + n_0)
    end

    return ξ, σ

end

@inline exp_inline(x) = @inline exp(x)

function gpd_quantile(p, ξ::T, sigma::T) where {T <: Real}
    return sigma * expm1(-ξ * log1p(-p)) / ξ
end

function psis_ess(
    weights::AbstractMatrix{T}, r_eff::AbstractVector{T}
) where T<:Real
    exp_entropy = zeros(T, size(weights, 1))
    @inbounds for y = axes(weights, 2), x = axes(weights, 1)
        exp_entropy[x] -= xlogx(weights[x, y])
    end
    for i = eachindex(exp_entropy)
        exp_entropy[i] = exp_inline(exp_entropy[i])
    end
    return r_eff .* exp_entropy
end


function psis_ess(weights::AbstractMatrix{<:Real})
    @warn "PSIS ESS not adjusted based on MCMC ESS. MCSE and ESS estimates " *
          "will be overoptimistic if samples are autocorrelated."
    return psis_ess(weights, ones(size(weights)))
end

function sup_ess(
    weights::AbstractMatrix{T}, r_eff::AbstractVector{T}
) where T<:Real
    return inv.(dropdims(maximum(weights; dims=2); dims=2)) .* r_eff
end



end