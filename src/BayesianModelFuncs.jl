using DataFrames, Distributed, Distributions, Turing, StatsFuns,  JLD2, Plots, Random, StatsBase, LaTeXStrings, CSV, StatsPlots, ReverseDiff, LinearAlgebra, ParetoSmooth
gr()

# ---- GLOBAL VARIABLES ---- #
stimContinuum = collect(range(-20, 40, length=9));

rawdata = DataFrame(CSV.File("../Exp2(lab)_forPub/Data/data.csv"))[:,:];

summaryData = combine(groupby(rawdata, [:subject, :votstep, :language]), :choseP => sum => :Obs_P, nrow => :N)

function subject_to_idx(data)
    d = Dict(subj => i for (i, subj) in enumerate(unique(data.subject)))
    data.subj_idx = [d[subj] for subj in data.subject]
    return data
end 

# summaryData = subject_to_idx(summaryData)

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

# ---- HELPER FUNCS ---- #
function dviz(d::Distribution) # convenience function for visualizing distributions
    tmp = rand(d,10000)
    if typeof(d) <: Beta
        display(histogram(tmp, xlims=(0,1)))
    else
        display(histogram(tmp))
    end
    
    return mean(tmp), std(tmp)
end

# ---- MODEL SPECS ---- #
@model function mod4cats(data)
    # ---- Priors ---- #
    logit_πₑ ~ Normal() # NOTE: worked well with just Normal(), so need to check if more biased prior is helpful or harmful
    πₑ := logistic(logit_πₑ)

    μ ~ MvNormal([0.0, 40.0, -40.0, 0.0], 10^2 * I)
    bₑ_μ := μ[1]  # Mean of /b/ in English
    pₑ_μ := μ[2]  # Mean of /p/ in English
    bₛ_μ := μ[3]  # Mean of /b/ in Spanish
    pₛ_μ := μ[4]  # Mean of /p/ in Spanish

    σ ~ truncated(Normal(17.5,5); lower=0) # Mean SD around 15–20ms
    σ_τ_cat ~ truncated(Cauchy(); lower=0)
    bₑ_σ ~ truncated(Normal(σ, σ_τ_cat); lower=0) # Gamma(5,2) # 
    pₑ_σ ~ truncated(Normal(σ, σ_τ_cat); lower=0)
    bₛ_σ ~ truncated(Normal(σ, σ_τ_cat); lower=0)
    pₛ_σ ~ truncated(Normal(σ, σ_τ_cat); lower=0)

    # ---- Likelihood ---- #
    x = stimContinuum[data.votstep]

    b_GMM = MixtureModel(Normal[
        Normal(bₑ_μ, bₑ_σ),
        Normal(bₛ_μ, bₛ_σ)], [πₑ, 1 - πₑ])
    log_b = logpdf(b_GMM, x)

    p_GMM = MixtureModel(Normal[
        Normal(pₑ_μ, pₑ_σ),
        Normal(pₛ_μ, pₛ_σ)], [πₑ, 1 - πₑ])
    log_p = logpdf(p_GMM, x)
    
    prob_p = clamp.(logistic.(log_p .- log_b), 1e-6, 1 - 1e-6)
    
    data.Obs_P .~ Binomial.(data.N, prob_p)
end

@model function mod2cats(data)
    # ---- Priors ---- #
    μ ~ MvNormal([0.0, 40.0], 10^2 * I)
    bₑ_μ := μ[1]  # Mean of /b/ in English
    pₑ_μ := μ[2]  # Mean of /p/ in English

    σ ~ truncated(Normal(17.5, 5); lower=1) # Mean SD around 15–20ms
    σ_τ_cat ~ truncated(Cauchy(); lower=0)
    bₑ_σ ~ truncated(Normal(σ, σ_τ_cat); lower=1)
    pₑ_σ ~ truncated(Normal(σ, σ_τ_cat); lower=1)

    # ---- Likelihood ---- #
    x = stimContinuum[data.votstep]
    log_b = logpdf(Normal(bₑ_μ, bₑ_σ), x)
    log_p = logpdf(Normal(pₑ_μ, pₑ_σ), x)

    prob_p = clamp.(logistic.(log_p .- log_b), 1e-6, 1 - 1e-6)
    
    data.Obs_P ~ arraydist([Binomial(data.N[i], prob_p[data.votstep[i]]) for i in eachindex(data.N)])
end

@model function mod4cats_hier(data)
    n_groups = length(unique(data.language))
    n_subjects = length(unique(data.subject))
    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]

    # region ---- ** PRIORS ** ---- 
        # region -- * MIXING WEIGHT * --
            # region GROUP LEVEL
                logit_πₑ_grp ~ filldist(Normal(), n_groups) # Mean of πₑ , biased towards just using English categories
                πₑ_grp := logistic.(logit_πₑ_grp) #
            # end

            # region SUBJECT LEVEL
                πₑ_τ_sub ~ truncated(Cauchy(); lower=0) # between-subject variability of πₑ
                logit_πₑ_sub ~ arraydist([Normal(logit_πₑ_grp[g], πₑ_τ_sub) for g in g_subj]) # Mean of /b/ in English for each subject
                πₑ_sub := logistic.(logit_πₑ_sub)
            # end
        # end

        # region -- * CATEGORY MEANS * -- 
            # region GROUP LEVEL
                #=
                    μ_τ = 10
                    ρ = 0.0
                    R = Matrix{Float64}(I, 4, 4)  # Start with identity matrix
                    R[[2,5,12,15]] .= ρ  # Set off-diagonal elements to ρ
                    Σ = μ_τ^2 * R
                =#
                μ_grp ~ filldist(MvNormal([0.0, 40.0, -40.0, 0.0], 10^2 * I), n_groups)
                bₑ_μ_grp := μ_grp[1, :]  # Mean of /b/ in English
                pₑ_μ_grp := μ_grp[2, :]  # Mean of /p/ in English
                bₛ_μ_grp := μ_grp[3, :]  # Mean of /b/ in Spanish
                pₛ_μ_grp := μ_grp[4, :]  # Mean of /p/ in Spanish
            # end

            # region SUBJECT LEVEL
                μ_τ_sub ~ truncated(Cauchy(); lower=0) # between-subject variability of category means  
                μ_sub ~ arraydist([MvNormal([bₑ_μ_grp[g], pₑ_μ_grp[g], bₛ_μ_grp[g], pₛ_μ_grp[g]], μ_τ_sub^2 * I) for g in g_subj]) # Mean of /b/ and /p/ in English and Spanish for each subject
                bₑ_μ_sub := μ_sub[1,:]  # Mean of /b/ in English for each subject
                pₑ_μ_sub := μ_sub[2,:]  # Mean of /p/ in English for each subject
                bₛ_μ_sub := μ_sub[3,:]  # Mean of /b/ in Spanish for each subject
                pₛ_μ_sub := μ_sub[4,:]  # Mean of /p/ in Spanish for each subject
            # end
        # end

        # region -- * CATEGORY SDs * -- 
            # region GROUP LEVEL
                # σ_grp ~ filldist(truncated(Normal(17.5, 5); lower=0), n_groups) # Mean SD around 15–20ms
                # σ_τ_cat ~ truncated(Cauchy(); lower=0)
                # bₑ_σ_grp ~ arraydist([truncated(Normal(σ_grp[g], σ_τ_cat); lower=1) for g in 1:n_groups]) 
                # pₑ_σ_grp ~ arraydist([truncated(Normal(σ_grp[g], σ_τ_cat); lower=1) for g in 1:n_groups])
                # bₛ_σ_grp ~ arraydist([truncated(Normal(σ_grp[g], σ_τ_cat); lower=1) for g in 1:n_groups])
                # pₛ_σ_grp ~ arraydist([truncated(Normal(σ_grp[g], σ_τ_cat); lower=1) for g in 1:n_groups])

                bₑ_σ_grp ~ filldist(Uniform(1, 30), n_groups)
                pₑ_σ_grp ~ filldist(Uniform(1, 30), n_groups)
                bₛ_σ_grp ~ filldist(Uniform(1, 30), n_groups)
                pₛ_σ_grp ~ filldist(Uniform(1, 30), n_groups)
            # end

            # region SUBJECT LEVEL
                σ_τ_sub ~ truncated(Cauchy(); lower=0) 
                bₑ_σ_sub ~ arraydist([truncated(Normal(bₑ_σ_grp[g], σ_τ_sub); lower=1) for g in g_subj]) 
                pₑ_σ_sub ~ arraydist([truncated(Normal(pₑ_σ_grp[g], σ_τ_sub); lower=1) for g in g_subj])
                bₛ_σ_sub ~ arraydist([truncated(Normal(bₑ_σ_grp[g], σ_τ_sub); lower=1) for g in g_subj]) 
                pₛ_σ_sub ~ arraydist([truncated(Normal(bₑ_σ_grp[g], σ_τ_sub); lower=1) for g in g_subj]) 
            # end
        # end
    # end

    # region ---- ** LIKELIHOOD ** ---- #
        x = stimContinuum[data.votstep]

        logpdfs_p = hcat(
            [log(πₑ_sub[s]) + logpdf(Normal(pₑ_μ_sub[s], pₑ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)],
            [log(1 - πₑ_sub[s]) + logpdf(Normal(pₛ_μ_sub[s], pₛ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)]
        )
        log_p = logsumexp(logpdfs_p, dims=2)[:, 1]  # dims=2: across columns for each row
        
        logpdfs_b = hcat(
            [log(πₑ_sub[s]) + logpdf(Normal(bₑ_μ_sub[s], bₑ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)],
            [log(1 - πₑ_sub[s]) + logpdf(Normal(bₛ_μ_sub[s], bₛ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)]
        )
        log_b = logsumexp(logpdfs_b, dims=2)[:, 1]  # dims=2: across columns for each row

        prob_p = clamp.(1.0 ./ (1 .+ exp.(log_b .- log_p)), 1e-6, 1 - 1e-6)
        
        data.Obs_P .~ Binomial.(data.N, prob_p)
    # end
end

@model function mod4cats_hier2(data)
    n_groups = length(unique(data.language))
    n_subjects = length(unique(data.subject))
    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]

    # region ---- ** PRIORS ** ---- 
        # region -- * MIXING WEIGHT * --
            # region GROUP LEVEL
                logit_πₑ_grp ~ filldist(Normal(), n_groups) # Mean of πₑ , biased towards just using English categories
                πₑ_grp := logistic.(logit_πₑ_grp) #
            # end

            # region SUBJECT LEVEL
                πₑ_τ_sub ~ truncated(Cauchy(); lower=0) # between-subject variability of πₑ
                logit_πₑ_sub ~ arraydist([Normal(logit_πₑ_grp[g], πₑ_τ_sub) for g in g_subj]) # Mean of /b/ in English for each subject
                πₑ_sub := logistic.(logit_πₑ_sub)
            # end
        # end

        # region -- * CATEGORY MEANS * -- 
            # region SUBJECT LEVEL
                μ_sub ~ filldist(MvNormal([0.0, 40.0, -40.0, 0.0], 10^2 * I), n_subjects) # Mean of /b/ and /p/ in English and Spanish for each subject
                bₑ_μ_sub := μ_sub[1,:]  # Mean of /b/ in English for each subject
                pₑ_μ_sub := μ_sub[2,:]  # Mean of /p/ in English for each subject
                bₛ_μ_sub := μ_sub[3,:]  # Mean of /b/ in Spanish for each subject
                pₛ_μ_sub := μ_sub[4,:]  # Mean of /p/ in Spanish for each subject
            # end
        # end

        # region -- * CATEGORY SDs * -- 
            # region SUBJECT LEVEL
                # σ ~ filldist(truncated(Normal(5,1);lower=0), n_subjects) # ~ truncated(Cauchy(); lower=0) # Mean SD around 15–20ms
                # σ_τ ~ filldist(truncated(Cauchy(); lower=0), n_subjects)
                # bₑ_σ_sub ~ arraydist([truncated(Normal(σ[i], σ_τ[i]); lower=0) for i in 1:n_subjects]) 
                # pₑ_σ_sub ~ arraydist([truncated(Normal(σ[i], σ_τ[i]); lower=0) for i in 1:n_subjects])
                # bₛ_σ_sub ~ arraydist([truncated(Normal(σ[i], σ_τ[i]); lower=0) for i in 1:n_subjects])
                # pₛ_σ_sub ~ arraydist([truncated(Normal(σ[i], σ_τ[i]); lower=0) for i in 1:n_subjects])

                α = 5 # ~ Gamma()
                β = 1 #~ Gamma()
                bₑ_σ_sub ~ filldist(Gamma(α,β), n_subjects) #filldist(truncated(Normal(σ, σ_τ_cat); lower=0), n_subjects)
                pₑ_σ_sub ~ filldist(Gamma(α,β), n_subjects)
                bₛ_σ_sub ~ filldist(Gamma(α,β), n_subjects)
                pₛ_σ_sub ~ filldist(Gamma(α,β), n_subjects)

            # end
        # end
    # end

    # region ---- ** LIKELIHOOD ** ---- #
        x = stimContinuum[data.votstep]

        logpdfs_p = hcat(
            [log(πₑ_sub[s]) + logpdf(Normal(pₑ_μ_sub[s], pₑ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)],
            [log(1 - πₑ_sub[s]) + logpdf(Normal(pₛ_μ_sub[s], pₛ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)]
        )
        log_p = logsumexp(logpdfs_p, dims=2)[:, 1]  # dims=2: across columns for each row
        
        logpdfs_b = hcat(
            [log(πₑ_sub[s]) + logpdf(Normal(bₑ_μ_sub[s], bₑ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)],
            [log(1 - πₑ_sub[s]) + logpdf(Normal(bₛ_μ_sub[s], bₛ_σ_sub[s]), x[i]) for (i, s) in enumerate(data.subj_idx)]
        )
        log_b = logsumexp(logpdfs_b, dims=2)[:, 1]  # dims=2: across columns for each row

        prob_p = clamp.(1.0 ./ (1 .+ exp.(log_b .- log_p)), 1e-6, 1 - 1e-6)
        
        data.Obs_P .~ Binomial.(data.N, prob_p)
    # end
end

@model function mod2cats_hier(data)
    n_groups = length(unique(data.language))
    n_subjects = length(unique(data.subject))
    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]
     
    # region ---- ** PRIORS ** ---- 
        # region -- * CATEGORY MEANS * -- 
            # region GROUP LEVEL
                #=
                    μ_τ = 10
                    ρ = 0.0
                    R = Matrix{Float64}(I, 2, 2)  # Start with identity matrix
                    R[[2,3]] .= ρ  # Set off-diagonal elements to ρ
                    Σ = μ_τ^2 * R
                =#
                μ_grp ~ filldist(MvNormal([0.0, 40.0], 10^2 * I), n_groups) # μ ~ MvNormal([0.0, 40.0], 10^2 * I) #[1 ρ;ρ 1] * I)
                bₑ_μ_grp := μ_grp[1, :]  # Mean of /b/ in English
                pₑ_μ_grp := μ_grp[2, :]  # Mean of /p/ in English
            # end

            # region SUBJECT LEVEL
                μ_τ_sub ~ truncated(Cauchy(); lower=0) # between-subject variability of category means
                μ_sub ~ arraydist([MvNormal([bₑ_μ_grp[g], pₑ_μ_grp[g]], μ_τ_sub^2 * I) for g in g_subj]) # Mean of /b/ and /p/ in English and Spanish for each subject
                bₑ_μ_sub := μ_sub[1,:]  # Mean of /b/ in English for each subject
                pₑ_μ_sub := μ_sub[2,:]  # Mean of /p/ in English for each subject
            # end
        # end

        # region -- * CATEGORY SDs * -- 
            # region GROUP LEVEL
                σ_grp ~ filldist(truncated(Normal(17.5,5); lower=0), n_groups) # Mean SD around 15–20ms
                σ_τ_cat ~ truncated(Cauchy(); lower=0)
                bₑ_σ_grp ~ arraydist([truncated(Normal(σ_grp[g], σ_τ_cat); lower=0) for g in 1:n_groups]) #(truncated(Normal(σ, σ_τ_cat); lower=1), n_groups) # SD for each category, same for all languages
                pₑ_σ_grp ~ arraydist([truncated(Normal(σ_grp[g], σ_τ_cat); lower=0) for g in 1:n_groups])
            # end

            # region SUBJECT LEVEL
                σ_τ_sub ~ truncated(Cauchy(); lower=0) # between-subject variability of category SD
                bₑ_σ_sub ~ arraydist([truncated(Normal(bₑ_σ_grp[g], σ_τ_sub); lower=0) for g in g_subj]) # SD for /b/ in English
                pₑ_σ_sub ~ arraydist([truncated(Normal(pₑ_σ_grp[g], σ_τ_sub); lower=0) for g in g_subj])

                # σ_τ_sub ~ truncated(Normal(); lower=0)
                # bₑ_σ_sub ~ arraydist([Gamma((bₑ_σ_grp[g]^2)/(σ_τ_sub^2), (σ_τ_sub^2)/bₑ_σ_grp[g]) for g in g_subj]) # SD for /b/ in English
                # pₑ_σ_sub ~ arraydist([Gamma((pₑ_σ_grp[g]^2)/(σ_τ_sub^2), (σ_τ_sub^2)/pₑ_σ_grp[g]) for g in g_subj]) # SD for /p/ in English

            # end
        # end
    # end

    # region ---- ** LIKELIHOOD ** ---- #
        x = stimContinuum[data.votstep]

        log_p = [logpdf(Normal(pₑ_μ_sub[s], pₑ_σ_sub[s]), x[i]) for (i,s) in enumerate(data.subj_idx)]  # dims=2: across columns for each row

        log_b = [logpdf(Normal(bₑ_μ_sub[s], bₑ_σ_sub[s]), x[i]) for (i,s) in enumerate(data.subj_idx)]  # dims=2: across columns for each row

        prob_p = clamp.(1.0 ./ (1 .+ exp.(log_b .- log_p)), 1e-6, 1 - 1e-6)
        
        data.Obs_P .~ Binomial.(data.N, prob_p)
    # end
end

## Model comparison funcs
function pointwise_loglik(chain, data, stimContinuum)
    version = (Symbol("πₑ_grp") in names(chain, :parameters) || Symbol("πₑ_grp[1]") in names(chain, :parameters) || Symbol("πₑ") in names(chain, :parameters)) ? "4cat" : "2cat"

    nsamples = length(chain) * size(chain,3) #length(chain[:bₑ_μ])  # number of posterior samples
    loglik = zeros(nsamples, length(data.Obs_P))  # Use the first subject for the analysis

    c = DataFrame(chain)

    for s in 1:nsamples
        if version == "4cat"
            if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
                i = unique(data.subj_idx)[1]
                πₑ = c[s,"πₑ_sub[$i]"]
                bₑ_μ = c[s,"bₑ_μ_sub[$i]"]
                pₑ_μ = c[s,"pₑ_μ_sub[$i]"]
                bₛ_μ = c[s,"bₛ_μ_sub[$i]"]
                pₛ_μ = c[s,"pₛ_μ_sub[$i]"]
                bₑ_σ = c[s,"bₑ_σ_sub[$i]"]
                pₑ_σ = c[s,"pₑ_σ_sub[$i]"]
                bₛ_σ = c[s,"bₛ_σ_sub[$i]"]
                pₛ_σ = c[s,"pₛ_σ_sub[$i]"]
            else
                πₑ = c[s,"πₑ"]
                bₑ_μ = c[s,"bₑ_μ"]
                pₑ_μ = c[s,"pₑ_μ"]
                bₛ_μ = c[s,"bₛ_μ"]
                pₛ_μ = c[s,"pₛ_μ"]
                bₑ_σ = c[s,"bₑ_σ"]
                pₑ_σ = c[s,"pₑ_σ"]
                bₛ_σ = c[s,"bₛ_σ"]
                pₛ_σ = c[s,"pₛ_σ"]
            end

            # # Define GMMs
            b_GMM = MixtureModel(Normal[
                Normal(bₑ_μ, bₑ_σ),
                Normal(bₛ_μ, bₛ_σ)], 
                [πₑ, 1 - πₑ]
            )

            p_GMM = MixtureModel(Normal[
                Normal(pₑ_μ, pₑ_σ),
                Normal(pₛ_μ, pₛ_σ)], 
                [πₑ, 1 - πₑ]
            )

            for j in 1:length(data.Obs_P)
                x = stimContinuum[data.votstep[j]]
                log_b = logpdf(b_GMM, x)
                log_p = logpdf(p_GMM, x)
                prob_p = clamp(1 / (1 + exp(log_b - log_p)), 1e-6, 1 - 1e-6)
                loglik[s, j] = logpdf(Binomial(data.N[j], prob_p), data.Obs_P[j])
            end
        else
            if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
                i = unique(data.subj_idx)[1]
                πₑ = 1.0
                bₑ_μ = c[s,"bₑ_μ_sub[$i]"]
                pₑ_μ = c[s,"pₑ_μ_sub[$i]"]
                bₛ_μ = -65.0
                pₛ_μ = 0.0
                bₑ_σ = c[s,"bₑ_σ_sub[$i]"]
                pₑ_σ = c[s,"pₑ_σ_sub[$i]"]
                bₛ_σ = 17.5
                pₛ_σ = 17.5
            else
                πₑ = 1.0
                bₑ_μ = c[s,"bₑ_μ"]
                pₑ_μ = c[s,"pₑ_μ"]
                bₛ_μ = -65.0
                pₛ_μ = 0.0
                bₑ_σ = c[s,"bₑ_σ"]
                pₑ_σ = c[s,"pₑ_σ"]
                bₛ_σ = 17.5
                pₛ_σ = 17.5
            end

            for j in 1:length(data.Obs_P)
                x = stimContinuum[data.votstep[j]]
                log_b = logpdf(Normal(bₑ_μ, bₑ_σ), x)
                log_p = logpdf(Normal(pₑ_μ, pₑ_σ), x)
                prob_p = clamp(1 / (1 + exp(log_b - log_p)), 1e-6, 1 - 1e-6)
                loglik[s, j] = logpdf(Binomial(data.N[j], prob_p), data.Obs_P[j])
            end
        end
    end

    return loglik
end

# subj = 14
# data = subject_to_idx(summaryData)#[summaryData.language .== "Monolingual English", :]
# curdata = data[data.subject .== subj, :]

# # ll_2cats = mean(sum(pointwise_loglik(chn_2cats, curdata, stimContinuum), dims=2))
# # ll_4cats = mean(sum(pointwise_loglik(chn_4cats, curdata, stimContinuum), dims=2))

# # BIC_2cats = -2 * ll_2cats + 2 * 4 # 6 parameters for 2cat model
# # BIC_4cats = -2 * ll_4cats + 2 * 9

# tur_ptwll = Turing.pointwise_loglikelihoods(mod4cats_hier(data), chn_4cats)
# my_ptwll = pointwise_loglik(chn_4cats, data, stimContinuum)
# par_ptwll = ParetoSmooth.pointwise_log_likelihoods

# tur1 = tur_ptwll["data.Obs_P[1]"]
# my1 = my_ptwll[:,1]

# res4 = psis_loo(mod_4cats, chn_4cats);
# res2 = psis_loo(mod2cats_hier(data), chn_2cats);

function compute_waic(ptw_ll)
    lppd = sum(log.(mean(exp.(ptw_ll), dims=1)))
    p_waic = sum(var(ptw_ll, dims=1))
    waic = -2 * (lppd - p_waic)
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

        b_GMM = MixtureModel(Normal[
            Normal(bₑ_μ, bₑ_σ),
            Normal(bₛ_μ, bₛ_σ)
        ], [πₑ, 1 - πₑ])
        log_b = logpdf(b_GMM, x)

        p_GMM = MixtureModel(Normal[
            Normal(pₑ_μ, pₑ_σ),
            Normal(pₛ_μ, pₛ_σ)
        ], [πₑ, 1 - πₑ])
        log_p = logpdf(p_GMM, x)

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
            bₛ_σ = 17.5
            pₛ_σ = 17.5
        else
            πₑ = 1.0
            bₑ_μ = mean(chain["bₑ_μ"])
            pₑ_μ = mean(chain["pₑ_μ"])
            bₛ_μ = -65.0
            pₛ_μ = 0.0
            bₑ_σ = mean(chain["bₑ_σ"])
            pₑ_σ = mean(chain["pₑ_σ"])
            bₛ_σ = 17.5
            pₛ_σ = 17.5
        end

        log_b = logpdf(Normal(bₑ_μ, bₑ_σ), x)
        log_p = logpdf(Normal(pₑ_μ, pₑ_σ), x)
    end

    xmin = [-20, bₑ_μ, bₛ_μ, pₑ_μ, pₛ_μ] |> minimum
    xmax = [40, bₑ_μ, bₛ_μ, pₑ_μ, pₛ_μ] |> maximum
    newStimContinuum = xmin:xmax
    
    prob_p = clamp.(1.0 ./ (1 .+ exp.(log_b .- log_p)), 1e-6, 1 - 1e-6)

    #default(fontfamily="Computer Modern")
    layout = @layout [a; b]
    fig = plot(layout=layout, size=(500,800), left_margin=5Plots.mm)

    # Top panel: observed vs predicted
    # loglik = round(mean(loglikelihood(model, chain)), digits=2)
    #legtitle_txt = L"$LL$: $\textbf{%$loglik}$"
    
    ptw_ll = pointwise_loglik(chain, curdata, stimContinuum);
    waic = round(compute_waic(ptw_ll),digits=2)
    legtitle_txt = L"$WAIC$: $\textbf{%$waic}$" 
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
        linewidth=2,
        ylim=(0,1),
        xlim=(xmin, xmax)
    )

    plot!(fig[1], stimContinuum, prob_p,
          label="Predicted",
          color=:black,
          linestyle =:dash,
          linewidth=2)

    ##
    eng_b = pdf.(Normal(bₑ_μ, bₑ_σ), newStimContinuum) .* πₑ
    eng_p = pdf.(Normal(pₑ_μ, pₑ_σ), newStimContinuum) .* πₑ

    spn_b = pdf.(Normal(bₛ_μ, bₛ_σ), newStimContinuum) .* (1-πₑ)
    spn_p = pdf.(Normal(pₛ_μ, pₛ_σ), newStimContinuum) .* (1-πₑ)
    
    wt = round(πₑ, digits=2)
    legtitle_txt = L"$ENG. Wt.$: $\textbf{%$wt}$"

    label_b_eng = L"\mathrm{/b/_{ENG}} \sim N(%$(Int(round(bₑ_μ))),%$(Int(round(bₑ_σ))))" #"Eng /b/; N($(Int(round(bₑ_μ))),$(Int(round(bₑ_σ))))"
    plot!(fig[2], newStimContinuum, eng_b, label=label_b_eng, color=:blue, linewidth=2, xlim=(xmin, xmax), 
            legend =:topleft, 
            background_color_legend = RGBA(0,0,0,.1),
            legendtitle = version == "4cat" ? legtitle_txt : nothing, 
            legendtitlefontsize=8,
            legend_font_pointsize=8,
        )
    
    label_p_eng = L"\mathrm{/p/_{ENG}} \sim N(%$(Int(round(pₑ_μ))),%$(Int(round(pₑ_σ))))"
    plot!(fig[2], newStimContinuum, eng_p, label=label_p_eng, color=:red, linewidth=2)
    
    if version == "4cat"
        label_b_spn = L"\mathrm{/b/_{SPN}} \sim N(%$(Int(round(bₛ_μ))),%$(Int(round(bₛ_σ))))"
        plot!(fig[2], newStimContinuum, spn_b, label=label_b_spn, color=:blue, linestyle=:dash, linewidth=2)
        
        label_p_spn = L"\mathrm{/p/_{SPN}} \sim N(%$(Int(round(pₛ_μ))),%$(Int(round(pₛ_σ))))"
        plot!(fig[2], newStimContinuum, spn_p, label=label_p_spn, color=:red, linestyle=:dash, linewidth=2)
    end
    xlabel!(fig[2], "VOT (ms)")
    ylabel!(fig[2], "PDF")
    title!(fig[2], "Category Distributions")

    return fig, waic
end