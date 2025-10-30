module BayesianModelFuncs 
# ================== START MODULE ==================
export mod2cats,
    mod4cats,
    mod2cats_hier,
    mod4cats_hier,
    plotFit,
    dviz,
    stimContinuum,
    group_map,
    lang_map,
    prepare_data_cat,
    getChnDFs4plot

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

function subject_to_idx(df)
    d = Dict(s => i for (i, s) in enumerate(unique(df.subject)))
    df.S = [d[s] for s in df.subject]
    return df
end 

function prepare_data_cat(df; subsample = false)
    if subsample 
        println("Subsampling 10 subjects from each group for quicker testing")
        Random.seed!(1)
        sampled_subjects = @chain df begin
            @select(subject, lang_grp)
            @distinct()
            @group_by(lang_grp)
            @slice_sample(n = 10, replace=false)
            @ungroup()
            @arrange(subject)
            @pull(subject)
        end
        df = @chain df @filter(subject in !!sampled_subjects)
    end;

    df = @chain df @mutate(G = case_when(lang_grp == "BE" => 1, lang_grp == "BS" => 2, lang_grp == "ME" => 3))
    Vstats = @chain df @group_by(G) @summarize(V̄ = mean(VOT), σV = std(VOT)) @arrange(G)
    df = @chain df begin
        @group_by(G)
        @mutate(Vz  = (VOT .- mean(VOT)) ./ std(VOT))
        @ungroup
        @group_by(G, lang_grp, subject, VOT, Vz)
        @summarize(Obs_P = sum(choseP), N = n())
        @ungroup
        @select(subject, lang_grp, G, VOT, Vz, Obs_P, N)
        @arrange(G, subject, Vz)
    end;

    df = subject_to_idx(df)

    S = df.S;
    G = @chain df @group_by(S) @slice(1) @ungroup() @pull(G);
    V = df.VOT;
    N = df.N;
    Y = df.Obs_P;
    
    return S, G, V, N, Y, df, Vstats
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
    logit_w0 ~ Normal(logit(.8), 1)
    w := logistic(logit_w0)

    μ ~ MvNormal([0.0, 40.0, -40.0, 0.0], 10^2 * I)
    bₑ_μ := μ[1]; pₑ_μ := μ[2]; bₛ_μ := μ[3]; pₛ_μ := μ[4]  # Mean of /p/ in Spanish

    σ0 ~ Normal(7.5, 2.5)
    σ_τ_cat ~ truncated(Normal(0,2.5); lower=0) 
    σ_cat_z ~ MvNormal(zeros(4), I)
    σ_cat := softplus.(σ0 .+ σ_τ_cat .* σ_cat_z) .+ 1e-3
    bₑ_σ := σ_cat[1]; pₑ_σ := σ_cat[2]; bₛ_σ := σ_cat[3]; pₛ_σ := σ_cat[4];

    # ---- Likelihood ---- #
    x = stimContinuum[data.votstep]
        
    logw = log(w)
    logπₛ = log(1 - w)
    logpdf_bₑ = logw .+ logpdf.(Normal(bₑ_μ, bₑ_σ), x)
    logpdf_pₑ = logw .+ logpdf.(Normal(pₑ_μ, pₑ_σ), x)
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
    V::AbstractVector{<:Real}, # VOT value for each row of the data
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
    S::AbstractVector{<:Integer}, # Subject index for each row of the data
    G::AbstractVector{<:Integer}, # Group index for each subject
    V::AbstractVector{<:Real}, # VOT value for each row of the data
    N::AbstractVector{<:Integer}, # Number of observations for each subject x VOT combination
    Y::AbstractVector{<:Integer}) # Number of /p/ categorizations for each subject x VOT combination

    n_groups = length(unique(G))
    n_subjects = length(unique(S))

    # region ---- **** PRIORS **** ---- 

        # region -- *** MIXING WEIGHT *** --
            # region -- ** GROUP LEVEL ** -- 
                logitw_grp ~ MvNormal(fill(logit(.8), n_groups), I)
                w_grp := logistic.(logitw_grp)
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τw_sub ~ truncated(Normal(0,1); lower=0) 
                    logitw_sub_z ~ MvNormal(zeros(n_subjects), I)
                    logitw_sub := logitw_grp[G] .+ τw_sub .* logitw_sub_z  
                    w_sub := logistic.(logitw_sub)
                # end
        # end

        # region -- *** CATEGORY MEANS *** -- 
            bₑ0 ~ Normal(0.0, 10.0); bₛ0 ~ Normal(-40.0, 10.0)
            logΔpₑ0 ~ Normal(log(40.0), 0.25); logΔpₛ0 ~ Normal(log(40.0), 0.25);
            Δpₑ0 := exp(logΔpₑ0); Δpₛ0 := exp(logΔpₛ0);
            pₑ0 := bₑ0 + Δpₑ0; pₛ0 := bₛ0 + Δpₛ0;
            # region -- ** GROUP LEVEL ** -- 
                τbₑ_grp ~ truncated(Normal(0, 7.5); lower=0) 
                τbₛ_grp ~ truncated(Normal(0, 7.5); lower=0)
                τlogΔpₑ_grp ~ truncated(Normal(0,0.2); lower=0)
                τlogΔpₛ_grp ~ truncated(Normal(0,0.2); lower=0)
                
                bₑ_grp_z ~ MvNormal(zeros(n_groups), I)
                bₛ_grp_z ~ MvNormal(zeros(n_groups), I)
                logΔpₑ_grp_z ~ MvNormal(zeros(n_groups), I)
                logΔpₛ_grp_z ~ MvNormal(zeros(n_groups), I)

                bₑ_grp := bₑ0 .+ τbₑ_grp .* bₑ_grp_z
                bₛ_grp := bₛ0 .+ τbₛ_grp .* bₛ_grp_z
                logΔpₑ_grp := logΔpₑ0 .+ τlogΔpₑ_grp .* logΔpₑ_grp_z
                logΔpₛ_grp := logΔpₛ0 .+ τlogΔpₛ_grp .* logΔpₛ_grp_z
                Δpₑ_grp := exp.(logΔpₑ_grp); Δpₛ_grp := exp.(logΔpₛ_grp)
                pₑ_grp := bₑ_grp .+ Δpₑ_grp; pₛ_grp := bₛ_grp .+ Δpₛ_grp
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τbₑ_sub ~ truncated(Normal(0, 7.5); lower=0) 
                    τbₛ_sub ~ truncated(Normal(0, 7.5); lower=0)
                    τlogΔpₑ_sub ~ truncated(Normal(0,0.2); lower=0)
                    τlogΔpₛ_sub ~ truncated(Normal(0,0.2); lower=0)
                    
                    bₑ_sub_z ~ MvNormal(zeros(n_subjects), I)
                    bₛ_sub_z ~ MvNormal(zeros(n_subjects), I)
                    logΔpₑ_sub_z ~ MvNormal(zeros(n_subjects), I)
                    logΔpₛ_sub_z ~ MvNormal(zeros(n_subjects), I)

                    bₑ_sub := bₑ_grp[G] .+ τbₑ_sub .* bₑ_sub_z
                    bₛ_sub := bₛ_grp[G] .+ τbₛ_sub .* bₛ_sub_z
                    logΔpₑ_sub := logΔpₑ_grp[G] .+ τlogΔpₑ_sub .* logΔpₑ_sub_z
                    logΔpₛ_sub := logΔpₛ_grp[G] .+ τlogΔpₛ_sub .* logΔpₛ_sub_z
                    Δpₑ_sub := exp.(logΔpₑ_sub); Δpₛ_sub := exp.(logΔpₛ_sub)
                    pₑ_sub := bₑ_sub .+ Δpₑ_sub; pₛ_sub := bₛ_sub .+ Δpₛ_sub
                # end
        # end

        # region -- *** CATEGORY SDs *** -- 
            logσ0 ~ filldist(Normal(log(10), .5), 4) 
            # region -- ** GROUP LEVEL ** -- 
                τlogσ_grp ~ filldist(truncated(Normal(0, .4); lower=0.0), 4) 
                logσ_grp_z ~ filldist(MvNormal(zeros(4), I), n_groups)
                logσ_grp := logσ0 .+ τlogσ_grp .* logσ_grp_z
                σ_grp := exp.(logσ_grp)
                σbₑ_grp := σ_grp[1, :]; σpₑ_grp := σ_grp[2, :]; σbₛ_grp := σ_grp[3, :]; σpₛ_grp := σ_grp[4, :]
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τlogσ_sub ~ filldist(truncated(Normal(0, .3); lower=0.0), 4) # log-scale subject SD
                    logσ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)
                    logσ_sub := logσ_grp[:, G] .+ τlogσ_sub .* logσ_sub_z
                    σ_sub := exp.(logσ_sub)
                    σbₑ_sub := σ_sub[1, :]; σpₑ_sub := σ_sub[2, :]; σbₛ_sub := σ_sub[3, :]; σpₛ_sub := σ_sub[4, :]
                # end
        # end
    # end

    # region ---- **** LIKELIHOOD **** ---- #
        logwₑ = log.(w_sub[S])
        logwₛ = log.(1 .- w_sub[S])
        logpdf_bₑ = logwₑ .+ logpdf.(Normal.(bₑ_sub[S], σbₑ_sub[S]), V)
        logpdf_pₑ = logwₑ .+ logpdf.(Normal.(pₑ_sub[S], σpₑ_sub[S]), V)
        logpdf_bₛ = logwₛ .+ logpdf.(Normal.(bₛ_sub[S], σbₛ_sub[S]), V)
        logpdf_pₛ = logwₛ .+ logpdf.(Normal.(pₛ_sub[S], σpₛ_sub[S]), V)
        log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
        log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)

        Y ~ product_distribution(Binomial.(N, prob_p))
        # for i in 1:nrow(data)
        #     data.Obs_P[i] ~ Binomial(data.N[i], prob_p[i])
        # end
    # end
end

## Model comparison funcs
function pointwise_loglik(chain, data, stimContinuum)
    version = (Symbol("w_grp") in names(chain, :parameters) || Symbol("w_grp[1]") in names(chain, :parameters) || Symbol("w") in names(chain, :parameters)) ? "4cat" : "2cat"

    idxs = data.subj_idx
    if version == "4cat"
        w = permutedims(cat([chain["w_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2))#;
        bₑ_μ = permutedims(cat([chain["bₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_μ = permutedims(cat([chain["pₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₛ_μ = permutedims(cat([chain["bₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₛ_μ = permutedims(cat([chain["pₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₑ_σ = permutedims(cat([chain["σbₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_σ = permutedims(cat([chain["σpₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₛ_σ = permutedims(cat([chain["σbₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₛ_σ = permutedims(cat([chain["σpₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));

        x = stimContinuum[data.votstep]

        logpdf_bₑ = log.(w) .+ logpdf.(Normal.(bₑ_μ, bₑ_σ), x);
        logpdf_pₑ = log.(w) .+ logpdf.(Normal.(pₑ_μ, pₑ_σ), x);
        logpdf_bₛ = log.(1 .- w) .+ logpdf.(Normal.(bₛ_μ, bₛ_σ), x);
        logpdf_pₛ = log.(1 .- w) .+ logpdf.(Normal.(pₛ_μ, pₛ_σ), x);

        log_p = log.(exp.(logpdf_pₑ) .+ exp.(logpdf_pₛ));
        log_b = log.(exp.(logpdf_bₑ) .+ exp.(logpdf_bₛ));
        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12);
        logliks = logpdf.(Binomial.(data.N, prob_p), data.Obs_P)

    else
        bₑ_μ = permutedims(cat([chain["bₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_μ = permutedims(cat([chain["pₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₑ_σ = permutedims(cat([chain["σbₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_σ = permutedims(cat([chain["σpₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));

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

function getChnDFs4plot(chn, df, Vstats)
    chndf = DataFrame(summarize(chn))
    subdf = @chain chndf begin
        @filter(occursin(r"(_sub)", String(parameters)))
        @filter(occursin(r"\[(\d+)\]$", String(parameters)))
        @filter(!occursin(r"(logit)", String(parameters)))
        @filter(!occursin(r"(_z)", String(parameters)))
        @filter(!occursin(r"(τ)", String(parameters)))
    end 
    transform!(subdf, 
        :parameters => ByRow(p -> begin
            s = String(p)                           # convert Symbol → String
            m = match(r"\[(\d+)\]$", s)             # find trailing [N]
            m === nothing ? missing : parse(Int, m.captures[1])
        end) => :S)
    transform!(subdf, :parameters => ByRow(p -> replace(String(p), r"\[\d+\]$" => "")) => :param)
    subdf = unstack(subdf, :S, :param, :mean)
    GS = @chain df @select(G, S) @group_by(S) @slice(1) @ungroup
    subdf = @left_join(GS, subdf, S)
    # subdf = @left_join(subdf, Vstats, G)
    # subdf = @chain subdf begin
    #     @mutate(
    #         bₑ_sub = bₑ_sub * σV + V̄, σbₑ_sub = σbₑ_sub * σV,
    #         pₑ_sub = pₑ_sub * σV + V̄, σpₑ_sub = σpₑ_sub * σV,
    #         bₛ_sub = bₛ_sub * σV + V̄, σbₛ_sub = σbₛ_sub * σV,
    #         pₛ_sub = pₛ_sub * σV + V̄, σpₛ_sub = σpₛ_sub * σV
    #     )
    # end

    return subdf
end

## Plotting funcs
function plotFit(chain, data, subj)
    version = (Symbol("w_grp") in names(chain, :parameters) || Symbol("w_grp[1]") in names(chain, :parameters) || Symbol("w") in names(chain, :parameters)) ? "4cat" : "2cat"
    
    curdata = data[data.subject .== subj, :]
    lang = lang_map[unique(curdata.lang_grp)[1]]
    g = group_map[curdata.lang_grp[1]]

    x = stimContinuum[curdata.votstep]
    if version == "4cat"
        #append_idx = ifelse(Symbol("bₑ_μ_sub[1]") in names(chain, :parameters), "[$i]", "")
        if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
            i = curdata.subj_idx[1]  # Use the first index to get the group
            w = mean(chain["w_sub[$i]"])
            bₑ_μ = mean(chain["bₑ_μ_sub[$i]"])
            pₑ_μ = mean(chain["pₑ_μ_sub[$i]"])
            bₛ_μ = mean(chain["bₛ_μ_sub[$i]"])
            pₛ_μ = mean(chain["pₛ_μ_sub[$i]"])
            bₑ_σ = mean(chain["bₑ_σ_sub[$i]"])
            pₑ_σ = mean(chain["pₑ_σ_sub[$i]"])
            bₛ_σ = mean(chain["bₛ_σ_sub[$i]"])
            pₛ_σ = mean(chain["pₛ_σ_sub[$i]"])
        else
            w = mean(chain["w"])
            bₑ_μ = mean(chain["bₑ_μ"])
            pₑ_μ = mean(chain["pₑ_μ"])
            bₛ_μ = mean(chain["bₛ_μ"])
            pₛ_μ = mean(chain["pₛ_μ"])
            bₑ_σ = mean(chain["bₑ_σ"])
            pₑ_σ = mean(chain["pₑ_σ"])
            bₛ_σ = mean(chain["bₛ_σ"])
            pₛ_σ = mean(chain["pₛ_σ"])
        end

        logw = log(w)
        logπₛ = log(1 - w)
        logpdf_bₑ = logw .+ logpdf.(Normal(bₑ_μ, bₑ_σ), x)
        logpdf_pₑ = logw .+ logpdf.(Normal(pₑ_μ, pₑ_σ), x)
        logpdf_bₛ = logπₛ .+ logpdf.(Normal(bₛ_μ, bₛ_σ), x)
        logpdf_pₛ = logπₛ .+ logpdf.(Normal(pₛ_μ, pₛ_σ), x)
        log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
        log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

    else
        if Symbol("bₑ_μ_sub[1]") in names(chain, :parameters)
            i = curdata.subj_idx[1]  # Use the first index to get the group
            w = 1.0
            bₑ_μ = mean(chain["bₑ_μ_sub[$i]"])
            pₑ_μ = mean(chain["pₑ_μ_sub[$i]"])
            bₛ_μ = -65.0
            pₛ_μ = 0.0
            bₑ_σ = mean(chain["bₑ_σ_sub[$i]"])
            pₑ_σ = mean(chain["pₑ_σ_sub[$i]"])
            bₛ_σ = 0
            pₛ_σ = 0
        else
            w = 1.0
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
    eng_b = pdf.(Normal(bₑ_μ, bₑ_σ), newStimContinuum) .* w
    eng_p = pdf.(Normal(pₑ_μ, pₑ_σ), newStimContinuum) .* w

    spn_b = pdf.(Normal(bₛ_μ, bₛ_σ), newStimContinuum) .* (1-w)
    spn_p = pdf.(Normal(pₛ_μ, pₛ_σ), newStimContinuum) .* (1-w)
    
    wt = round(w, digits=2)
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