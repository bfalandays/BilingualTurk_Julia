module CategoryModel 
# ================== START MODULE ==================

export mod2cats,
    mod4cats,
    mod2cats_hier,
    mod4cats_hier,
    logreg_hier,
    plotFit,
    plotFit_logreg,
    prepare_data_cat,
    getChnDFs

using Reexport
@reexport using ..Common, ..ParetoSmooth
using LaTeXStrings, LinearAlgebra
gr()

# ---- HELPER FUNCS ---- #
function prepare_data_cat(df; subsample = false)
    if subsample 
        println("Subsampling 10 subjects from each group for quicker testing")
        Random.seed!(2)
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

    df = @chain df @mutate(
        G = case_when(lang_grp == "BE" => 1, lang_grp == "BS" => 2, lang_grp == "ME" => 3),
        L = case_when(lang_grp == "BE" => 0, lang_grp == "BS" => 1, lang_grp == "ME" => 0),
        B = case_when(lang_grp == "BE" => 1, lang_grp == "BS" => 1, lang_grp == "ME" => 0))
    # Vstats = @chain df @group_by(G) @summarize(V̄ = mean(VOT), σV = std(VOT)) @arrange(G)
    df = @chain df begin
        @group_by(G)
        # @mutate(Vz  = (VOT .- mean(VOT)) ./ std(VOT))
        @ungroup
        @group_by(G, L, B, lang_grp, subject, VOT)
        @summarize(Obs_P = sum(choseP), N = n())
        @ungroup
        @select(subject, lang_grp, G, L, B, VOT, Obs_P, N)
        @arrange(G, subject, VOT)
    end;

    df = subject_to_idx(df)

    S = df.S;
    G = @chain df @group_by(S) @slice(1) @ungroup() @pull(G);
    L = @chain df @group_by(S) @slice(1) @ungroup() @pull(L);
    B = @chain df @group_by(S) @slice(1) @ungroup() @pull(B);
    V = df.VOT;
    N = df.N;
    Y = df.Obs_P;
    
    return S, G, L, B, V, N, Y, df
end

# ---- MODEL SPECS ---- #
@model function mod2cats(
    V::AbstractVector{<:Real}, # VOT value for each row of the data
    N::AbstractVector{<:Integer}, # Number of observations for each subject x VOT combination
    Y::AbstractVector{<:Integer}) # Number of /p/ categorizations for each subject x VOT combination
    
    # ---- Priors ---- #
    μₑ_sub ~ MvNormal([0.0, 40.0], 5^2 * I)
    bₑ_sub := μₑ_sub[1]; pₑ_sub := μₑ_sub[2];  

    σ_sub ~ Normal(7.5, 2.5)
    τσ_cat ~ truncated(Normal(0,2.5); lower=0) 
    σ_cat_z ~ MvNormal(zeros(2), I)
    σ_cat := softplus.(σ_sub .+ τσ_cat .* σ_cat_z) .+ 1e-3
    σbₑ_sub := σ_cat[1]; σpₑ_sub := σ_cat[2]; 

    # ---- Likelihood ---- #
    log_b = logpdf.(Normal(bₑ_sub, σbₑ_sub), V)
    log_p = logpdf.(Normal(pₑ_sub, σpₑ_sub), V)

    prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)
    
    Y ~ product_distribution(Binomial.(N, prob_p))
end

@model function mod4cats(
    V::AbstractVector{<:Real}, # VOT value for each row of the data
    N::AbstractVector{<:Integer}, # Number of observations for each subject x VOT combination
    Y::AbstractVector{<:Integer}) # Number of /p/ categorizations for each subject x VOT combination
    
    # ---- Priors ---- #
    # logitw_sub ~ Normal(logit(.95), .3)
    # w_sub := logistic(logitw_sub)
    # w_sub ~ Beta(30,1)
    w_sub := .99

    μ ~ MvNormal([0.0, 40.0, -40.0, 0.0], 2.5^2 * I)
    bₑ_sub := μ[1]; pₑ_sub := μ[2]; bₛ_sub := μ[3]; pₛ_sub := μ[4]  # Mean of /p/ in Spanish

    # μₑ_sub ~ MvNormal([0.0, 40.0], 2.5^2 * I)
    # logΔμₛ_sub ~ MvNormal(fill(log(40), 2), (.125^2) * I)
    # Δbₛ_sub := exp(logΔμₛ_sub[1]); Δpₛ_sub := exp(logΔμₛ_sub[2]);
    # bₑ_sub := μₑ_sub[1]; pₑ_sub := μₑ_sub[2];
    # bₛ_sub := bₑ_sub - Δbₛ_sub; pₛ_sub := pₑ_sub - Δpₛ_sub;

    # σ_cat ~ MvNormal(fill(2.5, 4), 5.0^2 * I)
    σ_cat ~ MvNormal(fill(2.5, 4), 5.0^2 * I)
    σbₑ_sub := softplus(σ_cat[1]) + 1e-3; σpₑ_sub := softplus(σ_cat[2]) + 1e-3; σbₛ_sub := softplus(σ_cat[3]) + 1e-3; σpₛ_sub := softplus(σ_cat[4]) + 1e-3;

    # ---- Likelihood ---- #
        
    logwₑ = log.(w_sub)
    logwₛ = log.(1 .- w_sub)
    logpdf_bₑ = logwₑ .+ logpdf.(Normal.(bₑ_sub, σbₑ_sub), V)
    logpdf_pₑ = logwₑ .+ logpdf.(Normal.(pₑ_sub, σpₑ_sub), V)
    logpdf_bₛ = logwₛ .+ logpdf.(Normal.(bₛ_sub, σbₛ_sub), V)
    logpdf_pₛ = logwₛ .+ logpdf.(Normal.(pₛ_sub, σpₛ_sub), V)
    log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
    log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

    prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)

    Y ~ product_distribution(Binomial.(N, prob_p))
    # for i in 1:nrow(data)
    #     data.Obs_P[i] ~ Binomial(data.N[i], prob_p[i])
    # end
end

@model function mod2cats_hier(
    S::AbstractVector{<:Integer}, # Subject index for each row of the data
    L::AbstractVector{<:Integer}, # language condition index for each subject
    B::AbstractVector{<:Integer}, # bilingualism index for each subject
    V::AbstractVector{<:Real}, # VOT value for each row of the data
    N::AbstractVector{<:Integer}, # Number of observations for each subject x VOT combination
    Y::AbstractVector{<:Integer}) # Number of /p/ categorizations for each subject x VOT combination

    # n_groups = length(unique(G))
    n_subjects = length(unique(S))

    # region ---- **** PRIORS **** ---- 

        # region -- *** CATEGORY MEANS *** -- 
            # region -- ** GROUP LEVEL ** -- 
                μ0 ~ MvNormal([0.0, 40.0], 2.5^2 * I)
                bₑ0 := μ0[1]; pₑ0 := μ0[2];
                
                βb_μ ~ MvNormal(zeros(2), 5.0^2 * I)
                βl_μ ~ MvNormal(zeros(2), 5.0^2 * I)

                bₑ_grp := [bₑ0 + βb_μ[1], bₑ0 + βb_μ[1] + βl_μ[1], bₑ0]; 
                pₑ_grp := [pₑ0 + βb_μ[2], pₑ0 + βb_μ[2] + βl_μ[2], pₑ0];
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τμ_sub ~ truncated(Normal(0, 2.5); lower=0) 
                    μ_sub_z ~ filldist(MvNormal(zeros(2), I), n_subjects)
                    
                    bₑ_sub := bₑ0 .+ (βb_μ[1] .* B) .+ (βl_μ[1] .* L) .+ τμ_sub .* μ_sub_z[1, :];
                    pₑ_sub := pₑ0 .+ (βb_μ[2] .* B) .+ (βl_μ[2] .* L) .+ τμ_sub .* μ_sub_z[2, :];
                # end
        # end

        # region -- *** CATEGORY SDs *** -- 
            # region -- ** GROUP LEVEL ** -- 

                σ_cat0 ~ MvNormal(fill(7.5, 2), 2.5^2 * I)
                σbₑ0 := σ_cat0[1]; σpₑ0 := σ_cat0[2]; 

                # βb_σ ~ MvNormal(zeros(2), 2.5^2 * I)
                # βl_σ ~ MvNormal(zeros(2), 2.5^2 * I)
                
                # σbₑ_grp := [σbₑ0 + βb_σ[1], σbₑ0 + βb_σ[1] + βl_σ[1], σbₑ0];
                # σpₑ_grp := [σpₑ0 + βb_σ[2], σpₑ0 + βb_σ[2] + βl_σ[2], σpₑ0];
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τσ_sub ~ truncated(Normal(0, 2.5); lower=0.0)
                    σ_sub_z ~ filldist(MvNormal(zeros(2), I), n_subjects)

                    σbₑ_sub := softplus.(σbₑ0 .+ τσ_sub .* σ_sub_z[1, :]) .+ 1e-3; 
                    σpₑ_sub := softplus.(σpₑ0 .+ τσ_sub .* σ_sub_z[2, :]) .+ 1e-3; 

                    # σbₑ_sub := softplus.(σbₑ0 .+ (βb_σ[1] .* B) .+ (βl_σ[1] .* L) .+ τσ_sub .* σ_sub_z[1, :]) .+ 1e-3; 
                    # σpₑ_sub := softplus.(σpₑ0 .+ (βb_σ[2] .* B) .+ (βl_σ[2] .* L) .+ τσ_sub .* σ_sub_z[2, :]) .+ 1e-3;
                # end
        # end
    # end

    # region ---- **** LIKELIHOOD **** ---- #
        log_b = logpdf.(Normal.(bₑ_sub[S], σbₑ_sub[S]), V)
        log_p = logpdf.(Normal.(pₑ_sub[S], σpₑ_sub[S]), V)

        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12)

        Y ~ product_distribution(Binomial.(N, prob_p))
    # end
end

@model function mod4cats_hier(
    S::AbstractVector{<:Integer}, # Subject index for each row of the data
    L::AbstractVector{<:Integer}, # language condition index for each subject
    B::AbstractVector{<:Integer}, # bilingualism index for each subject
    # M::AbstractVector{<:Real}, # ME group indicator for each subject
    V::AbstractVector{<:Real}, # VOT value for each row of the data
    N::AbstractVector{<:Integer}, # Number of observations for each subject x VOT combination
    Y::AbstractVector{<:Integer}) # Number of /p/ categorizations for each subject x VOT combination

    # n_groups = length(unique(G))
    n_subjects = length(unique(S))

    # region ---- **** PRIORS **** ---- 

        # region -- *** MIXING WEIGHT *** --
            # region -- ** GROUP LEVEL ** -- 
                logitw0 ~ Normal(logit(.5), 1.0)
                w0 := logistic.(logitw0)
                
                # w0 ~ Beta(30,1)
                # w0 := .99
                # logitw0 := logit(w0)

                βb_w ~ Normal(0, 2.5)
                βl_w ~ Normal(0, 2.5)
                
                w_grp := logistic.([logitw0 + βb_w, logitw0 + βb_w + βl_w, logitw0])
                # w_grp := .5 .+ .5 .* logistic.([logitw0 + βb_w, logitw0 + βb_w + βl_w, logitw0])
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τw_sub ~ truncated(Normal(0,1.0); lower=0) 
                    logitw_sub_z ~ MvNormal(zeros(n_subjects), I)
                    logitw_sub := logitw0 .+ (βb_w .* B) .+ (βl_w .* L) .+ τw_sub .* logitw_sub_z  
                    
                    w_sub := logistic.(logitw_sub)
                    # w_sub := .5 .+ .5 .* logistic.(logitw_sub)
                # end
        # end

        # region -- *** CATEGORY MEANS *** -- 
            # region -- ** GROUP LEVEL ** -- 
                μ0 ~ MvNormal([0.0, 40.0, -40.0, 0.0], 2.5^2 * I);
                bₑ0 := μ0[1]; pₑ0 := μ0[2]; bₛ0 := μ0[3]; pₛ0 := μ0[4];
                
                # βb_μ ~ MvNormal(zeros(4), 5.0^2 * I)
                # βl_μ ~ MvNormal(zeros(4), 5.0^2 * I)

                # bₑ_grp := [bₑ0 + βb_μ[1], bₑ0 + βb_μ[1] + βl_μ[1], bₑ0]; 
                # pₑ_grp := [pₑ0 + βb_μ[2], pₑ0 + βb_μ[2] + βl_μ[2], pₑ0];
                # bₛ_grp := [bₛ0 + βb_μ[3], bₛ0 + βb_μ[3] + βl_μ[3], bₛ0]; 
                # pₛ_grp := [pₛ0 + βb_μ[4], pₛ0 + βb_μ[4] + βl_μ[4], pₛ0];
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τμ_sub ~ truncated(Normal(0, 2.5); lower=0) 
                    μ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)
                    
                    # bₑ_sub := bₑ0 .+ (βb_μ[1] .* B) .+ (βl_μ[1] .* L) .+ τμ_sub .* μ_sub_z[1, :];
                    # pₑ_sub := pₑ0 .+ (βb_μ[2] .* B) .+ (βl_μ[2] .* L) .+ τμ_sub .* μ_sub_z[2, :];
                    # bₛ_sub := bₛ0 .+ (βb_μ[3] .* B) .+ (βl_μ[3] .* L) .+ τμ_sub .* μ_sub_z[3, :];
                    # pₛ_sub := pₛ0 .+ (βb_μ[4] .* B) .+ (βl_μ[4] .* L) .+ τμ_sub .* μ_sub_z[4, :];

                    bₑ_sub := bₑ0 .+ τμ_sub .* μ_sub_z[1, :];
                    pₑ_sub := pₑ0 .+ τμ_sub .* μ_sub_z[2, :];
                    bₛ_sub := bₛ0 .+ τμ_sub .* μ_sub_z[3, :];
                    pₛ_sub := pₛ0 .+ τμ_sub .* μ_sub_z[4, :];
                # end
        # end

        # region -- *** CATEGORY SDs *** -- 
            # region -- ** GROUP LEVEL ** -- 

                # σ_cat0 ~ MvNormal(fill(2.5, 4), 5.0^2 * I)
                σ_cat0 ~ MvNormal(fill(7.5, 4), 2.5^2 * I)
                σbₑ0 := σ_cat0[1]; σpₑ0 := σ_cat0[2]; 
                σbₛ0 := σ_cat0[3]; σpₛ0 := σ_cat0[4];

                # βb_σ ~ MvNormal(zeros(4), 2.5^2 * I)
                # βl_σ ~ MvNormal(zeros(4), 2.5^2 * I)
                
                # σbₑ_grp := [σbₑ0 + βb_σ[1], σbₑ0 + βb_σ[1] + βl_σ[1], σbₑ0];
                # σpₑ_grp := [σpₑ0 + βb_σ[2], σpₑ0 + βb_σ[2] + βl_σ[2], σpₑ0];
                # σbₛ_grp := [σbₛ0 + βb_σ[3], σbₛ0 + βb_σ[3] + βl_σ[3], σbₛ0]; 
                # σpₛ_grp := [σpₛ0 + βb_σ[4], σpₛ0 + βb_σ[4] + βl_σ[4], σpₛ0];
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τσ_sub ~ truncated(Normal(0, 2.5); lower=0.0)
                    σ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)

                    σbₑ_sub := softplus.(σbₑ0 .+ τσ_sub .* σ_sub_z[1, :]) .+ 1e-3; 
                    σpₑ_sub := softplus.(σpₑ0 .+ τσ_sub .* σ_sub_z[2, :]) .+ 1e-3; 
                    σbₛ_sub := softplus.(σbₛ0 .+ τσ_sub .* σ_sub_z[3, :]) .+ 1e-3; 
                    σpₛ_sub := softplus.(σpₛ0 .+ τσ_sub .* σ_sub_z[4, :]) .+ 1e-3;
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
    # end
end

function getversion(chn)
    if (Symbol("w_sub") in names(chn, :parameters))
        return "4cat", false
    elseif (Symbol("w_sub[1]") in names(chn, :parameters)) 
        return "4cat", true
    elseif (Symbol("bₑ_sub") in names(chn, :parameters))
        return "2cat", false
    else 
        return "2cat", true
    end
end

## Model comparison funcs
function pointwise_loglik(chn, curdata)
    version, hier = getversion(chn)

    idxs = curdata.S
    if version == "4cat"
        w = permutedims(cat([chn["w_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₑ_μ = permutedims(cat([chn["bₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_μ = permutedims(cat([chn["pₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₛ_μ = permutedims(cat([chn["bₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₛ_μ = permutedims(cat([chn["pₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₑ_σ = permutedims(cat([chn["σbₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_σ = permutedims(cat([chn["σpₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₛ_σ = permutedims(cat([chn["σbₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₛ_σ = permutedims(cat([chn["σpₛ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));

        x = curdata.VOT

        logpdf_bₑ = log.(w) .+ logpdf.(Normal.(bₑ_μ, bₑ_σ), x);
        logpdf_pₑ = log.(w) .+ logpdf.(Normal.(pₑ_μ, pₑ_σ), x);
        logpdf_bₛ = log.(1 .- w) .+ logpdf.(Normal.(bₛ_μ, bₛ_σ), x);
        logpdf_pₛ = log.(1 .- w) .+ logpdf.(Normal.(pₛ_μ, pₛ_σ), x);

        log_p = log.(exp.(logpdf_pₑ) .+ exp.(logpdf_pₛ));
        log_b = log.(exp.(logpdf_bₑ) .+ exp.(logpdf_bₛ));
        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12);
        logliks = logpdf.(Binomial.(curdata.N, prob_p), curdata.Obs_P);

    else
        bₑ_μ = permutedims(cat([chn["bₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_μ = permutedims(cat([chn["pₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        bₑ_σ = permutedims(cat([chn["σbₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));
        pₑ_σ = permutedims(cat([chn["σpₑ_sub[$i]"].data for i in idxs]...; dims=3), (3,1,2));

        x = curdata.VOT

        log_b = logpdf.(Normal.(bₑ_μ, bₑ_σ), x);
        log_p = logpdf.(Normal.(pₑ_μ, pₑ_σ), x);
        prob_p = clamp.(logistic.(log_p .- log_b), 1e-12, 1 - 1e-12);
        logliks = logpdf.(Binomial.(curdata.N, prob_p), curdata.Obs_P)
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

function getChnDFs(chn, df)
    chndf = DataFrame(summarize(chn))
    subdf = @chain chndf begin
        @filter(occursin(r"(_sub)", String(parameters)))
        @filter(occursin(r"\[(\d+)\]$", String(parameters)))
        @filter(!occursin(r"(log)", String(parameters)))
        @filter(!occursin(r"(Δ)", String(parameters)))
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

    grpdf = @chain chndf begin
        @filter(occursin(r"(_grp)", String(parameters)))
        @filter(occursin(r"\[(\d+)\]$", String(parameters)))
        @filter(!occursin(r"(log)", String(parameters)))
        @filter(!occursin(r"(Δ)", String(parameters)))
        @filter(!occursin(r"(_z)", String(parameters)))
        @filter(!occursin(r"(τ)", String(parameters)))
    end 
    transform!(grpdf, 
        :parameters => ByRow(p -> begin
            s = String(p)                           # convert Symbol → String
            m = match(r"\[(\d+)\]$", s)             # find trailing [N]
            m === nothing ? missing : parse(Int, m.captures[1])
        end) => :G)
    transform!(grpdf, :parameters => ByRow(p -> replace(String(p), r"\[\d+\]$" => "")) => :param)
    grpdf = unstack(grpdf, :G, :param, :mean)

    return chndf, subdf, grpdf
end

## Plotting funcs
function plotFit(chn, df, s)
    version, hier = getversion(chn)

    curdata = @chain df @filter(S == !!s)
    lang = unique(curdata.lang_grp)[1]
    g = unique(curdata.G) #group_map[curdata.lang_grp[1]]

    x = curdata.VOT
    if version == "4cat"
        #append_idx = ifelse(Symbol("bₑ_μ_sub[1]") in names(chn, :parameters), "[$i]", "")
        if hier
            w = mean(chn["w_sub[$s]"])
            bₑ_μ = mean(chn["bₑ_sub[$s]"])
            pₑ_μ = mean(chn["pₑ_sub[$s]"])
            bₛ_μ = mean(chn["bₛ_sub[$s]"])
            pₛ_μ = mean(chn["pₛ_sub[$s]"])
            bₑ_σ = mean(chn["σbₑ_sub[$s]"])
            pₑ_σ = mean(chn["σpₑ_sub[$s]"])
            bₛ_σ = mean(chn["σbₛ_sub[$s]"])
            pₛ_σ = mean(chn["σpₛ_sub[$s]"])
        else
            w = mean(chn["w_sub"])
            bₑ_μ = mean(chn["bₑ_sub"])
            pₑ_μ = mean(chn["pₑ_sub"])
            bₛ_μ = mean(chn["bₛ_sub"])
            pₛ_μ = mean(chn["pₛ_sub"])
            bₑ_σ = mean(chn["σbₑ_sub"])
            pₑ_σ = mean(chn["σpₑ_sub"])
            bₛ_σ = mean(chn["σbₛ_sub"])
            pₛ_σ = mean(chn["σpₛ_sub"])
        end

        logwₑ = log(w)
        logwₛ = log(1 - w)
        logpdf_bₑ = logwₑ .+ logpdf.(Normal(bₑ_μ, bₑ_σ), x)
        logpdf_pₑ = logwₑ .+ logpdf.(Normal(pₑ_μ, pₑ_σ), x)
        logpdf_bₛ = logwₛ .+ logpdf.(Normal(bₛ_μ, bₛ_σ), x)
        logpdf_pₛ = logwₛ .+ logpdf.(Normal(pₛ_μ, pₛ_σ), x)
        log_b = logsumexp(hcat(logpdf_bₑ, logpdf_bₛ), dims=2)[:, 1]
        log_p = logsumexp(hcat(logpdf_pₑ, logpdf_pₛ), dims=2)[:, 1]

    else
        if hier
            i = curdata.S[1]  # Use the first index to get the group
            w = 1.0
            bₑ_μ = mean(chn["bₑ_sub[$i]"])
            pₑ_μ = mean(chn["pₑ_sub[$i]"])
            bₛ_μ = -65.0
            pₛ_μ = 0.0
            bₑ_σ = mean(chn["σbₑ_sub[$i]"])
            pₑ_σ = mean(chn["σpₑ_sub[$i]"])
            bₛ_σ = 0
            pₛ_σ = 0
        else
            w = 1.0
            bₑ_μ = mean(chn["bₑ_sub"])
            pₑ_μ = mean(chn["pₑ_sub"])
            bₛ_μ = -65.0
            pₛ_μ = 0.0
            bₑ_σ = mean(chn["σbₑ_sub"])
            pₑ_σ = mean(chn["σpₑ_sub"])
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
    Brier = round(sum(Nᵢ .* (pᵢ .- p̂ᵢ).^2) / sum(Nᵢ),digits=3);
    #χ² = sum((Oᵢ .- Eᵢ) .^ 2 ./ (Eᵢ .* (1 .- p̂ᵢ)))
    ll_model = sum(@. Oᵢ * log(p̂ᵢ) + (Nᵢ - Oᵢ) * log(1 - p̂ᵢ)); 
    ll_saturated = sum(@. Oᵢ * log(pᵢ) + (Nᵢ - Oᵢ) * log(1 - pᵢ)); 
    D = -2 * (ll_model - ll_saturated) |> x->round(x, digits=2) # deviance statistic
    elpd, legtitle_txt =try
        pll = pointwise_loglik(chn, curdata); #waic = round(compute_waic(pll),digits=2)
        res = ParetoSmooth.psis_loo(pll; source=:mcmc)
        if !isnan(res.estimates(statistic=:cv_elpd, column=:total))
            elpd = round(res.estimates(statistic=:cv_elpd, column=:total), digits=2);
            (elpd, L"$Brier$: $\textbf{%$Brier}$; $ELPD$: $\textbf{%$elpd}$; $D$: $\textbf{%$D}$")
        else
            elpd = round(res.estimates(statistic=:naive_lpd, column=:total), digits=2);
            (elpd, L"$Brier$: $\textbf{%$Brier}$; $Naive LPD$: $\textbf{%$elpd}$; $D$: $\textbf{%$D}$")
        end
    catch e
        (NaN, L"$Brier$: $\textbf{%$Brier}$; $ELPD$: $\textbf{NaN}$; $D$: $\textbf{%$D}$")
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

######



# ================== END MODULE ==================
end