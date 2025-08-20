module BayesianModelFuncs 
# ================== START MODULE ==================
export mod2cats,
    mod4cats,
    mod2cats_hier,
    mod4cats_hier,
    plotFit,
    dviz,
    stimContinuum,
    subject_to_idx,
    group_map,
    lang_map

using Reexport
@reexport using DataFrames, TidierData, CSV, Random, StatsFuns, Distributed, Distributions, Turing, ParetoSmooth, ReverseDiff, Plots, LaTeXStrings, LinearAlgebra, JLD2
gr()

# ---- HELPER FUNCS ---- #
function dviz(d) # convenience function for visualizing distributions
    tmp = rand(d,10000)
    if typeof(d) <: Beta
        display(histogram(tmp, xlims=(0,1)))
    else
        display(histogram(tmp))
    end
    
    return mean(tmp), std(tmp)
end

function subject_to_idx(data)
    d = Dict(s => i for (i, s) in enumerate(unique(data.subject)))
    data.S = [d[s] for s in data.subject]
    return data
end 

stimContinuum = collect(range(-20, 40, length=9));

group_map = Dict(
    "Monolingual English" => 1,
    "Bilingual English" => 2,
    "Bilingual Spanish" => 3
)

lang_map = Dict(
        "Monolingual English" => "ME",
        "Bilingual English" => "BE",
        "Bilingual Spanish" => "BS"
)

# ---- MODEL SPECS ---- #
@model function mod2cats(data)
    # ---- Priors ---- #
    μ0 ~ MvNormal([0.0, 40.0], 10^2 * I)
    bₑ_μ := μ0[1]; pₑ_μ := μ0[2]  

    σ0 ~ Normal(7.5, 2.5)
    σ_τ_cat ~ truncated(Normal(0,2.5); lower=0) 
    σ_cat_z ~ MvNormal(zeros(2), I)
    σ_cat := softplus.(σ0 .+ σ_τ_cat .* σ_cat_z) .+ 1e-3
    bₑ_σ := σ_cat[1]; pₑ_σ := σ_cat[2]

    # ---- Likelihood ---- #
    x = stimContinuum[data.votstep]
    log_b = logpdf.(Normal(bₑ_μ, bₑ_σ), x)
    log_p = logpdf.(Normal(pₑ_μ, pₑ_σ), x)

    prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)
    
    data.Obs_P ~ product_distribution(Binomial.(data.N, prob_p))
    # for i in 1:nrow(data)
    #     data.Obs_P[i] ~ Binomial(data.N[i], prob_p[i])
    # end
end

@model function mod4cats(data)
    # ---- Priors ---- #
    logit_πₑ0 ~ Normal(logit(.8), 1)
    πₑ := logistic(logit_πₑ0)

    μ ~ MvNormal([0.0, 40.0, -40.0, 0.0], 10^2 * I)
    bₑ_μ := μ[1]; pₑ_μ := μ[2]; bₛ_μ := μ[3]; pₛ_μ := μ[4]  # Mean of /p/ in Spanish

    σ0 ~ Normal(7.5, 2.5)
    σ_τ_cat ~ truncated(Normal(0,2.5); lower=0) 
    σ_cat_z ~ MvNormal(zeros(4), I)
    σ_cat := softplus.(σ0 .+ σ_τ_cat .* σ_cat_z) .+ 1e-3
    bₑ_σ := σ_cat[1]; pₑ_σ := σ_cat[2]; bₛ_σ := σ_cat[3]; pₛ_σ := σ_cat[4];

    # ---- Likelihood ---- #
    x = stimContinuum[data.votstep]
        
    logπₑ = log(πₑ)
    logπₛ = log(1 - πₑ)
    logpdf_bₑ = logπₑ .+ logpdf.(Normal(bₑ_μ, bₑ_σ), x)
    logpdf_pₑ = logπₑ .+ logpdf.(Normal(pₑ_μ, pₑ_σ), x)
    logpdf_bₛ = logπₛ .+ logpdf.(Normal(bₛ_μ, bₛ_σ), x)
    logpdf_pₛ = logπₛ .+ logpdf.(Normal(pₛ_μ, pₛ_σ), x)
    log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
    log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

    prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)

    data.Obs_P ~ product_distribution(Binomial.(data.N, prob_p))
    # for i in 1:nrow(data)
    #     data.Obs_P[i] ~ Binomial(data.N[i], prob_p[i])
    # end
end

@model function mod2cats_hier(
    S::AbstractVector{<:Real}, # Subject index for each row of the data
    G::AbstractVector{<:Real}, # Group index for each subject
    V::AbstractVector{<:Real}, # VOT step for each row of the data
    N::AbstractVector{<:Real}, # Number of trials for each row of the data
    y::AbstractVector{<:Real}) # Observed proportions for each row of the data

    n_groups = length(unique(G))
    n_subjects = length(unique(S))

    # region ---- **** PRIORS **** ---- 

        # region -- *** CATEGORY MEANS *** -- 
            μ0 ~ MvNormal([0.0, 40.0], 5^2 * I)
            # region -- ** GROUP LEVEL ** -- 
                μ_τ_grp ~ filldist(truncated(Normal(0, 2.5); lower=0), 2)
                μ_grp_z ~ filldist(MvNormal(zeros(2), I), n_groups) 
                μ_grp := μ0 .+ μ_τ_grp .* μ_grp_z 
                bₑ_μ_grp := μ_grp[1,:]; pₑ_μ_grp := μ_grp[2,:]; 
            # end
                # region -- * SUBJECT LEVEL * -- 
                    μ_τ_sub ~ filldist(truncated(Normal(0, 2.5), lower=0), 2)
                    μ_sub_z ~ filldist(MvNormal(zeros(2), I), n_subjects)
                    μ_sub := μ_grp[:, G] .+ μ_τ_sub .* μ_sub_z 
                    bₑ_μ_sub := μ_sub[1, :]; pₑ_μ_sub := μ_sub[2, :]; 
                # end
        # end

        # region -- *** CATEGORY SDs *** -- 
            σ0 ~ filldist(truncated(Normal(10.0, 2.5); lower =1), 2) 
            # region -- ** GROUP LEVEL ** -- 
                σ_τ_grp ~ filldist(truncated(Normal(0, 1); lower=0), 2) 
                σ_grp_z ~ filldist(MvNormal(zeros(2), I), n_groups)
                σ_grp := softplus.(σ0 .+ σ_τ_grp .* σ_grp_z) .+ 1e-3 
                bₑ_σ_grp := σ_grp[1, :]; pₑ_σ_grp := σ_grp[2, :]; 
            # end
                # region -- * SUBJECT LEVEL * -- 
                    σ_τ_sub ~ filldist(truncated(Normal(0, 1); lower=0), 2) # log-scale subject SD
                    σ_sub_z ~ filldist(MvNormal(zeros(2), I), n_subjects)
                    σ_sub = softplus.(σ_grp[:, G] .+ σ_τ_sub .* σ_sub_z) .+ 1e-3    # 4 × n_subjects
                    bₑ_σ_sub := σ_sub[1, :]; pₑ_σ_sub := σ_sub[2, :]; 
                # end
        # end
    # end

    # region ---- **** LIKELIHOOD **** ---- #
        log_b = logpdf.(Normal.(bₑ_μ_sub[S], bₑ_σ_sub[S]), V)
        log_p = logpdf.(Normal.(pₑ_μ_sub[S], pₑ_σ_sub[S]), V)

        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)

        y ~ product_distribution(Binomial.(N, prob_p))
        # for i in 1:nrow(data)
        #     data.Obs_P[i] ~ Binomial(data.N[i], prob_p[i])
        # end
    # end
end

@model function mod4cats_hier(
    S::AbstractVector{<:Real}, # Subject index for each row of the data
    G::AbstractVector{<:Real}, # Group index for each subject
    V::AbstractVector{<:Real}, # VOT step for each row of the data
    N::AbstractVector{<:Real}, # Number of trials for each row of the data
    y::AbstractVector{<:Real}) # Observed proportions for each row of the data

    n_groups = length(unique(G))
    n_subjects = length(unique(S))

    # region ---- **** PRIORS **** ---- 

        # region -- *** MIXING WEIGHT *** --
            logit_πₑ0 ~ Normal(logit(.8), 1)
            πₑ0 := logistic(logit_πₑ0)
            # region -- ** GROUP LEVEL ** -- 
                πₑ_τ_grp ~ truncated(Normal(0, .5); lower=0)
                logit_πₑ_grp_z ~ MvNormal(zeros(n_groups), I) # logit_πₑ_grp_z ~ MvNormal(zeros(n_groups), I)
                logit_πₑ_grp := logit_πₑ0 .+  πₑ_τ_grp .* logit_πₑ_grp_z #~ filldist(Normal(logit_πₑ0, πₑ_τ_grp), n_groups)
                πₑ_grp := logistic.(logit_πₑ_grp)
            # end
                # region -- * SUBJECT LEVEL * -- 
                    πₑ_τ_sub ~ truncated(Normal(0,.5); lower=0) #filldist(truncated(Normal(0,.5); lower=0), n_groups) # between-subject variability of πₑ
                    logit_πₑ_sub_z ~ MvNormal(zeros(n_subjects), I)
                    logit_πₑ_sub := logit_πₑ_grp[G] .+ πₑ_τ_sub .* logit_πₑ_sub_z #~ MvNormal(logit_πₑ_grp[G], Diagonal(πₑ_τ_sub[G] .^ 2))  
                    πₑ_sub := logistic.(logit_πₑ_sub)
                # end
        # end

        # region -- *** CATEGORY MEANS *** -- 
            μ0 ~ MvNormal([0.0, 40.0, -40.0, 0.0], 5^2 * I)
            # region -- ** GROUP LEVEL ** -- 
                μ_τ_grp ~ filldist(truncated(Normal(0, 2.5); lower=0), 4)
                μ_grp_z ~ filldist(MvNormal(zeros(4), I), n_groups) # filldist(MvNormal(μ0, Diagonal(μ_τ_grp .^ 2)), n_groups)
                μ_grp := μ0 .+ μ_τ_grp .* μ_grp_z #~ filldist(MvNormal(μ0, Diagonal(μ_τ_grp .^ 2)), n_groups) # := μ0 .+ μ_τ_grp .* μ_grp_z                                  
                bₑ_μ_grp := μ_grp[1,:]; pₑ_μ_grp := μ_grp[2,:]; bₛ_μ_grp := μ_grp[3,:]; pₛ_μ_grp := μ_grp[4,:]
            # end
                # region -- * SUBJECT LEVEL * -- 
                    μ_τ_sub ~ filldist(truncated(Normal(0, 2.5), lower=0), 4)
                    μ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)
                    μ_sub := μ_grp[:, G] .+ μ_τ_sub .* μ_sub_z#~ arraydist([MvNormal(μ_grp[:, g], Diagonal(μ_τ_sub .^ 2)) for g in G])
                    bₑ_μ_sub := μ_sub[1, :]; pₑ_μ_sub := μ_sub[2, :]; bₛ_μ_sub := μ_sub[3, :]; pₛ_μ_sub := μ_sub[4, :]
                # end
        # end

        # region -- *** CATEGORY SDs *** -- 
            σ0 ~ filldist(truncated(Normal(10.0, 2.5); lower =1), 4) #filldist(Gamma(m^2/v, v/m), 4)
            # region -- ** GROUP LEVEL ** -- 
                σ_τ_grp ~ filldist(truncated(Normal(0, 1); lower=0), 4)#truncated(Normal(0, 1); lower=0)
                σ_grp_z ~ filldist(MvNormal(zeros(4), I), n_groups)
                σ_grp := softplus.(σ0 .+ σ_τ_grp .* σ_grp_z) .+ 1e-3 #softplus.(σ_cat .+ σ_τ_grp .* σ_grp_z) .+ 1e-3
                bₑ_σ_grp := σ_grp[1, :]; pₑ_σ_grp := σ_grp[2, :]; bₛ_σ_grp := σ_grp[3, :]; pₛ_σ_grp := σ_grp[4, :]
            # end
                # region -- * SUBJECT LEVEL * -- 
                    σ_τ_sub ~ filldist(truncated(Normal(0, 1); lower=0), 4) # log-scale subject SD
                    σ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)
                    σ_sub = softplus.(σ_grp[:, G] .+ σ_τ_sub .* σ_sub_z) .+ 1e-3    # 4 × n_subjects
                    bₑ_σ_sub := σ_sub[1, :]; pₑ_σ_sub := σ_sub[2, :]; bₛ_σ_sub := σ_sub[3, :]; pₛ_σ_sub := σ_sub[4, :]
                # end
        # end
    # end

    # region ---- **** LIKELIHOOD **** ---- #
        logπₑ = log.(πₑ_sub[S])
        logπₛ = log.(1 .- πₑ_sub[S])
        logpdf_bₑ = logπₑ .+ logpdf.(Normal.(bₑ_μ_sub[S], bₑ_σ_sub[S]), V)
        logpdf_pₑ = logπₑ .+ logpdf.(Normal.(pₑ_μ_sub[S], pₑ_σ_sub[S]), V)
        logpdf_bₛ = logπₛ .+ logpdf.(Normal.(bₛ_μ_sub[S], bₛ_σ_sub[S]), V)
        logpdf_pₛ = logπₛ .+ logpdf.(Normal.(pₛ_μ_sub[S], pₛ_σ_sub[S]), V)
        log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
        log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)

        y ~ product_distribution(Binomial.(N, prob_p))
        # for i in 1:nrow(data)
        #     data.Obs_P[i] ~ Binomial(data.N[i], prob_p[i])
        # end
    # end
end

@model function mod4cats_hier_reg(data)
    n_groups = length(unique(data.language))
    n_subjects = length(unique(data.subject))
    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]

    # region ---- ** PRIORS ** ---- 
        # region -- * MIXING WEIGHT * --
                # GRAND INTERCEPT
                β₀_logitπ ~ Normal(logit(4/5), 1.0) 

                # GROUP EFFECTS
                βⱼ_logitπ_raw ~ filldist(Normal(), n_groups)
                βⱼ_logitπ = βⱼ_logitπ_raw .- mean(βⱼ_logitπ_raw)
                logit_πⱼ = β₀_logitπ .+ βⱼ_logitπ
                πⱼ := logistic.(logit_πⱼ)

                # SUBJECT RANDOM INTERCEPTS
                τ_logitπ_i ~ truncated(Normal(0, 0.5); lower=0)
                zᵢ_logitπ ~ filldist(Normal(), n_subjects)
                uᵢ_logitπ = τ_logitπ_i .* zᵢ_logitπ

                logit_πᵢ = β₀_logitπ .+ βⱼ_logitπ[g_subj] .+ uᵢ_logitπ
                πᵢ := logistic.(logit_πᵢ)
        # end

        # region -- * CATEGORY MEANS * -- 
                # GRAND INTERCEPT
                β₀_μ ~ MvNormal([0.0, 40.0, -40.0, 0.0], 5^2 * I)

                # GROUP LEVEL
                τ_μ_j ~ filldist(truncated(Normal(0, 5); lower=0), 4)      # per-category between-group SD
                zⱼ_μ  ~ filldist(Normal(), 4, n_groups)                # 4 × n_groups standard normals
                βⱼ_μ_raw = τ_μ_j .* zⱼ_μ
                βⱼ_μ = βⱼ_μ_raw .- mean(βⱼ_μ_raw, dims=2)                  # enforce Σ_g βⱼ_μ[:,g] = 0
                μⱼ = β₀_μ .+ βⱼ_μ 

                # SUBJECT LEVEL
                τ_μ_i ~ filldist(truncated(Normal(0, 0.5); lower=0), 4)
                zᵢ_μ ~ filldist(MvNormal(zeros(4), I), n_subjects)
                uᵢ_μ = τ_μ_i .* zᵢ_μ
                μᵢ = β₀_μ .+ βⱼ_μ[:, g_subj] .+ uᵢ_μ

                # Expose group means 
                bₑ_μ_grp := μⱼ[1,:]; pₑ_μ_grp := μⱼ[2,:]; bₛ_μ_grp := μⱼ[3,:]; pₛ_μ_grp := μⱼ[4,:]
                # Expose subject means 
                bₑ_μ_sub := μᵢ[1,:]; pₑ_μ_sub := μᵢ[2,:]; bₛ_μ_sub := μᵢ[3,:]; pₛ_μ_sub := μᵢ[4,:]
        # end

        # region -- * CATEGORY SDs * -- 
            # GRAND INTERCEPT
            β₀_logσ ~ MvNormal(fill(log(5.0), 4), 0.3^2 * I)

            # GROUP LEVEL
            τ_logσ_j ~ filldist(truncated(Normal(0, 0.10); lower=0), 4)    # between-group SD per category on log scale
            zⱼ_logσ  ~ filldist(Normal(), (4, n_groups))                   # 4 × n_groups
            βⱼ_logσ_raw = τ_logσ_j .* zⱼ_logσ
            βⱼ_logσ = βⱼ_logσ_raw .- mean(βⱼ_logσ_raw, dims=2)             # Σ_g βⱼ_logσ[:,g] = 0
            logσⱼ = β₀_logσ .+ βⱼ_logσ 
            σⱼ = exp.(logσⱼ)

            # SUBJECT LEVEL
            τ_logσ_i ~ filldist(truncated(Normal(0, 0.10); lower=0), 4)     # between-subject SD per category
            zᵢ_logσ  ~ filldist(Normal(), (4, n_subjects))                  # 4 × n_subjects
            uᵢ_logσ = τ_logσ_i .* zᵢ_logσ                                   # subject deviations (mean 0)

            logσᵢ = β₀_logσ .+ βⱼ_logσ[:, g_subj] .+ uᵢ_logσ
            σᵢ = exp.(logσᵢ)

            # Expose group SDs
            bₑ_σ_grp := σⱼ[1,:]; pₑ_σ_grp := σⱼ[2,:]; bₛ_σ_grp := σⱼ[3,:]; pₛ_σ_grp := σⱼ[4,:]
            # Expose subject SDs
            bₑ_σ_sub := σᵢ[1,:]; pₑ_σ_sub := σᵢ[2,:]; bₛ_σ_sub := σᵢ[3,:]; pₛ_σ_sub := σᵢ[4,:]
        # end
    # end

    # region ---- ** LIKELIHOOD ** ---- #
        x = stimContinuum[data.votstep]

        logpdfs_p = hcat(
            [log(πᵢ[s]) + logpdf(Normal(pₑ_μ_sub[s], pₑ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)],
            [log(1 - πᵢ[s]) + logpdf(Normal(pₛ_μ_sub[s], pₛ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)]
        )
        log_p = logsumexp(logpdfs_p, dims=2)[:, 1]  # dims=2: across columns for each row
        
        logpdfs_b = hcat(
            [log(πᵢ[s]) + logpdf(Normal(bₑ_μ_sub[s], bₑ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)],
            [log(1 - πᵢ[s]) + logpdf(Normal(bₛ_μ_sub[s], bₛ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)]
        )
        log_b = logsumexp(logpdfs_b, dims=2)[:, 1]  # dims=2: across columns for each row

        prob_p = clamp.(1.0 ./ (1 .+ exp.(log_b .- log_p)), 1e-12, 1 - 1e-12)
        
        data.Obs_P ~ product_distribution(Binomial.(data.N, prob_p))
    # end
end

## Model comparison funcs
function pointwise_loglik(chain, data, stimContinuum)
    version = (Symbol("πₑ_grp") in names(chain, :parameters) || Symbol("πₑ_grp[1]") in names(chain, :parameters) || Symbol("πₑ") in names(chain, :parameters)) ? "4cat" : "2cat"

    idxs = data.subj_idx
    if version == "4cat"
        if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
            πₑ = permutedims(cat([chain["πₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2))#;
            bₑ_μ = permutedims(cat([chain["bₑ_μ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_μ = permutedims(cat([chain["pₑ_μ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            bₛ_μ = permutedims(cat([chain["bₛ_μ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            pₛ_μ = permutedims(cat([chain["pₛ_μ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            bₑ_σ = permutedims(cat([chain["bₑ_σ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_σ = permutedims(cat([chain["pₑ_σ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            bₛ_σ = permutedims(cat([chain["bₛ_σ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            pₛ_σ = permutedims(cat([chain["pₛ_σ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        else
            πₑ = permutedims(cat([chain["πₑ"].data for i in idxs]...; dims=3), (3,1,2));
            bₑ_μ = permutedims(cat([chain["bₑ_μ"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_μ = permutedims(cat([chain["pₑ_μ"].data for i in idxs]...; dims=3), (3,1,2));
            bₛ_μ = permutedims(cat([chain["bₛ_μ"].data for i in idxs]...; dims=3), (3,1,2));
            pₛ_μ = permutedims(cat([chain["pₛ_μ"].data for i in idxs]...; dims=3), (3,1,2));
            bₑ_σ = permutedims(cat([chain["bₑ_σ"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_σ = permutedims(cat([chain["pₑ_σ"].data for i in idxs]...; dims=3), (3,1,2));
            bₛ_σ = permutedims(cat([chain["bₛ_σ"].data for i in idxs]...; dims=3), (3,1,2));
            pₛ_σ = permutedims(cat([chain["pₛ_σ"].data for i in idxs]...; dims=3), (3,1,2));
        end

        x = stimContinuum[data.votstep]

        logpdf_bₑ = log.(πₑ) .+ logpdf.(Normal.(bₑ_μ, bₑ_σ), x);
        logpdf_pₑ = log.(πₑ) .+ logpdf.(Normal.(pₑ_μ, pₑ_σ), x);
        logpdf_bₛ = log.(1 .- πₑ) .+ logpdf.(Normal.(bₛ_μ, bₛ_σ), x);
        logpdf_pₛ = log.(1 .- πₑ) .+ logpdf.(Normal.(pₛ_μ, pₛ_σ), x);

        log_p = log.(exp.(logpdf_pₑ) .+ exp.(logpdf_pₛ));
        log_b = log.(exp.(logpdf_bₑ) .+ exp.(logpdf_bₛ));
        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12);
        logliks = logpdf.(Binomial.(data.N, prob_p), data.Obs_P)

    else
        if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
            bₑ_μ = permutedims(cat([chain["bₑ_μ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_μ = permutedims(cat([chain["pₑ_μ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            bₑ_σ = permutedims(cat([chain["bₑ_σ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_σ = permutedims(cat([chain["pₑ_σ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        else
            bₑ_μ = permutedims(cat([chain["bₑ_μ"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_μ = permutedims(cat([chain["pₑ_μ"].data for i in idxs]...; dims=3), (3,1,2));
            bₑ_σ = permutedims(cat([chain["bₑ_σ"].data for i in idxs]...; dims=3), (3,1,2));
            pₑ_σ = permutedims(cat([chain["pₑ_σ"].data for i in idxs]...; dims=3), (3,1,2));
        end

        x = stimContinuum[data.votstep]

        log_b = logpdf.(Normal.(bₑ_μ, bₑ_σ), x);
        log_p = logpdf.(Normal.(pₑ_μ, pₑ_σ), x);
        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12);
        logliks = logpdf.(Binomial.(data.N, prob_p), data.Obs_P)
    end

    return logliks
end

function compute_waic(pll)
    if ndims(pll) == 2
        lppd = sum(log.(mean(exp.(pll), dims=1)))
        p_waic = sum(var(pll, dims=1))
        waic = -2 * (lppd - p_waic)
    elseif ndims(pll) == 3
        lppd = sum(log.(mean(exp.(pll), dims=(2,3))))
        p_waic = sum(var(pll, dims=(2,3)))
        waic = -2 * (lppd - p_waic)
    else
        error("Invalid dimensions for pll: $(ndims(pll))")
    end
    
    return waic
end

## Plotting funcs
function plotFit(chain, data, subj)
    version = (Symbol("πₑ_grp") in names(chain, :parameters) || Symbol("πₑ_grp[1]") in names(chain, :parameters) || Symbol("πₑ") in names(chain, :parameters)) ? "4cat" : "2cat"
    
    curdata = data[data.subject .== subj, :]
    lang = lang_map[unique(curdata.language)[1]]
    g = group_map[curdata.language[1]]

    x = stimContinuum[curdata.votstep]
    if version == "4cat"
        #append_idx = ifelse(Symbol("bₑ_μ_sub[1]") in names(chain, :parameters), "[$i]", "")
        if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
            i = curdata.subj_idx[1]  # Use the first index to get the group
            πₑ = mean(chain["πₑ_sub[$i]"])
            bₑ_μ = mean(chain["bₑ_μ_sub[$i]"])
            pₑ_μ = mean(chain["pₑ_μ_sub[$i]"])
            bₛ_μ = mean(chain["bₛ_μ_sub[$i]"])
            pₛ_μ = mean(chain["pₛ_μ_sub[$i]"])
            bₑ_σ = mean(chain["bₑ_σ_sub[$i]"])
            pₑ_σ = mean(chain["pₑ_σ_sub[$i]"])
            bₛ_σ = mean(chain["bₛ_σ_sub[$i]"])
            pₛ_σ = mean(chain["pₛ_σ_sub[$i]"])
        else
            πₑ = mean(chain["πₑ"])
            bₑ_μ = mean(chain["bₑ_μ"])
            pₑ_μ = mean(chain["pₑ_μ"])
            bₛ_μ = mean(chain["bₛ_μ"])
            pₛ_μ = mean(chain["pₛ_μ"])
            bₑ_σ = mean(chain["bₑ_σ"])
            pₑ_σ = mean(chain["pₑ_σ"])
            bₛ_σ = mean(chain["bₛ_σ"])
            pₛ_σ = mean(chain["pₛ_σ"])
        end

        logπₑ = log(πₑ)
        logπₛ = log(1 - πₑ)
        logpdf_bₑ = logπₑ .+ logpdf.(Normal(bₑ_μ, bₑ_σ), x)
        logpdf_pₑ = logπₑ .+ logpdf.(Normal(pₑ_μ, pₑ_σ), x)
        logpdf_bₛ = logπₛ .+ logpdf.(Normal(bₛ_μ, bₛ_σ), x)
        logpdf_pₛ = logπₛ .+ logpdf.(Normal(pₛ_μ, pₛ_σ), x)
        log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
        log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

    else
        if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
            i = curdata.subj_idx[1]  # Use the first index to get the group
            πₑ = 1.0
            bₑ_μ = mean(chain["bₑ_μ_sub[$i]"])
            pₑ_μ = mean(chain["pₑ_μ_sub[$i]"])
            bₛ_μ = -65.0
            pₛ_μ = 0.0
            bₑ_σ = mean(chain["bₑ_σ_sub[$i]"])
            pₑ_σ = mean(chain["pₑ_σ_sub[$i]"])
            bₛ_σ = 0
            pₛ_σ = 0
        else
            πₑ = 1.0
            bₑ_μ = mean(chain["bₑ_μ"])
            pₑ_μ = mean(chain["pₑ_μ"])
            bₛ_μ = -65.0
            pₛ_μ = 0.0
            bₑ_σ = mean(chain["bₑ_σ"])
            pₑ_σ = mean(chain["pₑ_σ"])
            bₛ_σ = 0
            pₛ_σ = 0
        end

        log_b = logpdf(Normal(bₑ_μ, bₑ_σ), x)
        log_p = logpdf(Normal(pₑ_μ, pₑ_σ), x)
    end
    prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)
    xmin = -20 #[-20, bₑ_μ, bₛ_μ, pₑ_μ, pₛ_μ] |> minimum
    xmax = 40 #[40, bₑ_μ, bₛ_μ, pₑ_μ, pₛ_μ] |> maximum
    newStimContinuum = xmin:xmax
    
    Oᵢ = curdata.Obs_P ; Nᵢ = curdata.N; Eᵢ = Nᵢ .* prob_p ; pᵢ = @. clamp(Oᵢ ./ Nᵢ, 1e-12, 1 - 1e-12) ; p̂ᵢ = prob_p; 
    #Brier = sum(Nᵢ .* (pᵢ .- p̂ᵢ).^2) / sum(Nᵢ)
    #χ² = sum((Oᵢ .- Eᵢ) .^ 2 ./ (Eᵢ .* (1 .- p̂ᵢ)))
    ll_model = sum(@. Oᵢ * log(p̂ᵢ) + (Nᵢ - Oᵢ) * log(1 - p̂ᵢ)); 
    ll_saturated = sum(@. Oᵢ * log(pᵢ) + (Nᵢ - Oᵢ) * log(1 - pᵢ)); 
    D = -2 * (ll_model - ll_saturated) |> x->round(x, digits=2)
    pll = pointwise_loglik(chain, curdata, stimContinuum); #waic = round(compute_waic(pll),digits=2)
    elpd, legtitle_txt =try
        res = psis_loo(pll; source=:mcmc)
        if !isnan(res.estimates(statistic=:cv_elpd, column=:total))
            elpd = round(res.estimates(statistic=:cv_elpd, column=:total), digits=2);
            (elpd, L"$ELPD$: $\textbf{%$elpd}$; $D$: $\textbf{%$D}$")
        else
            elpd = round(res.estimates(statistic=:naive_lpd, column=:total), digits=2);
            (elpd, L"$Naive LPD$: $\textbf{%$elpd}$; $D$: $\textbf{%$D}$")
        end
    catch e
        (NaN, L"$ELPD$: $\textbf{NaN}$; $D$: $\textbf{%$D}$")
    end
    
    layout = @layout [a; b];
    fig = plot(layout=layout, size=(420,720), left_margin=5Plots.mm);
    plot!(fig[1], stimContinuum, curdata.Obs_P ./ curdata.N,
        label="Observed",
        color=:black,
        xlabel="VOT (ms)",
        ylabel="Proportion /p/",
        title="Subject $(curdata.subject[1]) ($(lang))",
        legend=:topleft,
        background_color_legend = RGBA(0,0,0,.1),
        legendtitle = legtitle_txt, 
        # legendtitlefontsize=8,
        legend_font_pointsize=8,
        linewidth=1.5,
        ylim=(0,1),
        xlim=(xmin, xmax)
    )

    plot!(fig[1], stimContinuum, prob_p,
          label="Predicted",
          color=:black,
          linestyle =:dash,
          linewidth=1.5)

    ##
    eng_b = pdf.(Normal(bₑ_μ, bₑ_σ), newStimContinuum) .* πₑ
    eng_p = pdf.(Normal(pₑ_μ, pₑ_σ), newStimContinuum) .* πₑ

    spn_b = pdf.(Normal(bₛ_μ, bₛ_σ), newStimContinuum) .* (1-πₑ)
    spn_p = pdf.(Normal(pₛ_μ, pₛ_σ), newStimContinuum) .* (1-πₑ)
    
    wt = round(πₑ, digits=2)
    legtitle_txt = L"$ENG. Wt.$: $\textbf{%$wt}$"

    label_b_eng = L"\mathrm{/b/_{ENG}} \sim N(%$(Int(round(bₑ_μ))),%$(Int(round(bₑ_σ))))" #"Eng /b/; N($(Int(round(bₑ_μ))),$(Int(round(bₑ_σ))))"
    plot!(fig[2], newStimContinuum, eng_b, label=label_b_eng, color=:blue, linewidth=1.5, xlim=(xmin, xmax), 
            legend =:topleft, 
            background_color_legend = RGBA(0,0,0,.1),
            legendtitle = version == "4cat" ? legtitle_txt : nothing, 
            legendtitlefontsize=8,
            legend_font_pointsize=8,
        )
    
    label_p_eng = L"\mathrm{/p/_{ENG}} \sim N(%$(Int(round(pₑ_μ))),%$(Int(round(pₑ_σ))))"
    plot!(fig[2], newStimContinuum, eng_p, label=label_p_eng, color=:red, linewidth=1.5)
    
    if version == "4cat"
        label_b_spn = L"\mathrm{/b/_{SPN}} \sim N(%$(Int(round(bₛ_μ))),%$(Int(round(bₛ_σ))))"
        plot!(fig[2], newStimContinuum, spn_b, label=label_b_spn, color=:blue, linestyle=:dash, linewidth=1.5)
        
        label_p_spn = L"\mathrm{/p/_{SPN}} \sim N(%$(Int(round(pₛ_μ))),%$(Int(round(pₛ_σ))))"
        plot!(fig[2], newStimContinuum, spn_p, label=label_p_spn, color=:red, linestyle=:dash, linewidth=1.5)
    end
    xlabel!(fig[2], "VOT (ms)")
    ylabel!(fig[2], "PDF")
    title!(fig[2], "Category Distributions")

    return fig, elpd
end

# ================== END MODULE ==================
end