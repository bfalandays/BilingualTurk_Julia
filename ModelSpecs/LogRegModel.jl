module LogRegModel
# ================== START MODULE ==================

export logreg_hier,
    plotFit_logreg_subj,
    plotFit_logreg_grp,
    prepare_data_logreg,
    getBoundaryDF,
    getChnDFs

using Reexport
@reexport using ..Common, ..ParetoSmooth
using LaTeXStrings, LinearAlgebra
gr()

function prepare_data_logreg(df; subsample = false)
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

    df = @chain df @mutate(G = case_when(lang_grp == "BE" => 1, lang_grp == "BS" => 2, lang_grp == "ME" => 3))
    # Vstats = @chain df @group_by(G) @summarize(V̄ = mean(VOT), σV = std(VOT)) @arrange(G)
    df = @chain df begin
        @group_by(G)
        # @mutate(Vz  = (VOT .- mean(VOT)) ./ std(VOT))
        @ungroup
        @group_by(G, lang_grp, subject, VOT)
        @summarize(Obs_P = sum(choseP), N = n())
        @ungroup
        @select(subject, lang_grp, G, VOT, Obs_P, N)
        @arrange(G, subject, VOT)
    end;

    df = subject_to_idx(df)

    S = df.S;
    G = @chain df @group_by(S) @slice(1) @ungroup() @pull(G);
    V = df.VOT;
    N = df.N;
    Y = df.Obs_P;
    
    return S, G, V, N, Y, df
end

# logit𝓁 = rand(Normal(logit(.05), .5), 1000); 𝓁 = logistic.(logit𝓁)
# logδ = rand(Normal(log(6.0), 1.0),1000); δ = exp.(logδ)
# 𝒹 = logistic.(logit𝓁 .+ δ)
# histogram(𝒹)
# describe(𝒹)

@model function logreg_hier(
    S::AbstractVector{<:Integer}, # Subject index for each row of the data
    G::AbstractVector{<:Integer}, # Group index for each subject
    V::AbstractVector{<:Real}, # VOT value for each row of the data
    N::AbstractVector{<:Integer}, # Number of observations for each subject x VOT combination
    Y::AbstractVector{<:Integer}) # Number of /p/ categorizations for each subject x VOT combination
    # Y::AbstractVector{<:Union{Missing, <:Integer}}) 

    n_groups = length(unique(G))
    n_subjects = length(unique(S))

    # region ---- **** PRIORS **** ---- 
        # Lower asymptote: logit-normal
        logit𝓁0 ~ MvNormal(fill(logit(0.05), n_groups), .5^2 * I); 𝓁0 := logistic.(logit𝓁0)
        τlogit𝓁_sub ~ truncated(Normal(0,1); lower=0)

        # gap (δ) between lower and upper asymptote on logit scale: model logδ on ℝ, then δ = exp(logδ)
        logδ0 ~ MvNormal(fill(log(6.0), n_groups), .5^2 * I); 𝒹0 := logistic.(logit𝓁0 .+ exp.(logδ0))
        τlogδ_sub ~ truncated(Normal(0,1); lower=0)
        
        # gradiency parameter (β): log-normal prior
        logβ0 ~ MvNormal(fill(log(.2), n_groups), 1.0^2 * I); β0 := exp.(logβ0)
        τlogβ_sub ~ truncated(Normal(0,1); lower=0)

        # crossover point (𝒸): normal prior on VOT scale
        𝒸0 ~ MvNormal(fill(10.0, n_groups), 7.5^2 * I)
        τ𝒸_sub ~ truncated(Normal(0,1); lower=0)

        # region -- * SUBJECT LEVEL * -- 
            # lower asymptote (𝓁)
            logit𝓁_sub_z ~ MvNormal(zeros(n_subjects), I)
            logit𝓁_sub := logit𝓁0[G] .+ τlogit𝓁_sub .* logit𝓁_sub_z  
            𝓁_sub := logistic.(logit𝓁_sub)

            # gap (δ) / upper asymptote (𝒹)
            logδ_sub_z ~ MvNormal(zeros(n_subjects), I)
            logδ_sub := logδ0[G] .+ τlogδ_sub .* logδ_sub_z
            𝒹_sub := logistic.(logit𝓁_sub .+ exp.(logδ_sub))

            logβ_sub_z ~ MvNormal(zeros(n_subjects), I)
            logβ_sub := logβ0[G] .+ τlogβ_sub .* logβ_sub_z
            β_sub := exp.(logβ_sub)

            𝒸_sub_z ~ MvNormal(zeros(n_subjects), I)
            𝒸_sub := 𝒸0[G] .+ τ𝒸_sub .* 𝒸_sub_z
    # end

    # region ---- **** LIKELIHOOD **** ---- #
        # prob_p = @. 𝓁_sub[S] + (𝒹_sub[S] - 𝓁_sub[S]) * logistic(β_sub[S] * (V - 𝒸_sub[S])) 
        # Y ~ product_distribution(Binomial.(N, prob_p))
        
        for (i, s) in enumerate(S)
            prob_p = 𝓁_sub[s] + (𝒹_sub[s] - 𝓁_sub[s]) * logistic(β_sub[s] * (V[i] - 𝒸_sub[s])) 
            Y[i] ~ Binomial(N[i], prob_p)
        end
    # end
end

function V50(𝓁, 𝒹, β, 𝒸)
    (𝓁 < 0.5 < 𝒹) || return(NaN)
    q = (0.5 - 𝓁) / (𝒹 - 𝓁)
    𝒸 + logit(q) / β
end

function getBoundaryDF(chn, df)
    boundarydf = @chain DataFrame(summarize(group(chn, :𝒸_sub))) begin
        @rename(boundary=mean) 
        @mutate(S = row_number()) 
        @select(S, boundary) 
        @left_join(@chain df @select(S, subject, lang_grp) @distinct) 
        @relocate(subject, lang_grp, before=boundary)
        @left_join(@chain DataFrame(summarize(group(chn, :β_sub))) begin
            @rename(gradiency=mean)
            @mutate(S = row_number()) 
            @select(S, gradiency)
        end)
        @left_join(@chain DataFrame(summarize(group(chn, :𝓁_sub))) begin
            @rename(floor=mean)
            @mutate(S = row_number()) 
            @select(S, floor)
        end)
        @left_join(@chain DataFrame(summarize(group(chn, :𝒹_sub))) begin
            @rename(ceil=mean)
            @mutate(S = row_number()) 
            @select(S, ceil)
        end)
        @mutate(range = ceil - floor)
        @mutate(x50 = V50(floor, ceil, gradiency, boundary))
    end

    return boundarydf
end

function plotFit_logreg_subj(chn, df, s)

    curdata = @chain df @filter(S == !!s)
    lang = unique(curdata.lang_grp)[1]
    g = unique(curdata.G) #group_map[curdata.lang_grp[1]]

    # x = stimContinuum
    # 𝓁 = mean(chn["𝓁_sub[$s]"])
    # 𝒹 = mean(chn["𝒹_sub[$s]"])
    # r = 𝒹 - 𝓁
    # β = mean(chn["β_sub[$s]"])
    # 𝒸 = mean(chn["𝒸_sub[$s]"])
    # prob_p = @. 𝓁 + (𝒹 - 𝓁) * logistic(β * (x - 𝒸))
    # x50 = V50(𝓁, 𝒹, β, 𝒸)

    vals = get(chn, [:𝓁_sub, :𝒹_sub, :β_sub, :𝒸_sub]);

    l = vec(vals.𝓁_sub[s].data);
    d = vec(vals.𝒹_sub[s].data);
    b = vec(vals.β_sub[s].data);
    c = vec(vals.𝒸_sub[s].data);
    prob_ps = l .+ (d .- l) .* logistic.(b .* (stimContinuum' .- c));
    prob_p = mean(prob_ps; dims=1) |> vec;
    CIs = [quantile(prob_ps[:,i], [0.025, 0.975]) for i in 1:9]; CIs_lo = [ci[1] for ci in CIs]; CIs_hi = [ci[2] for ci in CIs];
    x50 = V50.(l, d, b, c) |> mean;
    
    Oᵢ = curdata.Obs_P ; Nᵢ = curdata.N; Eᵢ = Nᵢ .* prob_p ; pᵢ = @. clamp(Oᵢ ./ Nᵢ, 1e-12, 1 - 1e-12) ; p̂ᵢ = prob_p; 
    ll_model = sum(@. Oᵢ * log(p̂ᵢ) + (Nᵢ - Oᵢ) * log(1 - p̂ᵢ)); 
    ll_saturated = sum(@. Oᵢ * log(pᵢ) + (Nᵢ - Oᵢ) * log(1 - pᵢ)); 
    D = -2 * (ll_model - ll_saturated) |> x->round(x, digits=2) # deviance statistic
    Brier = round(sum(Nᵢ .* (pᵢ .- p̂ᵢ).^2) / sum(Nᵢ), digits=3)
    legtitle_txt = L"$Brier$: $\textbf{%$Brier}$; $D$: $\textbf{%$D}$"

    palette = [:red, :green, :blue]

    fig = plot(size=(420,360), left_margin=5Plots.mm);
    plot!(fig, stimContinuum, curdata.Obs_P ./ curdata.N,
        label="Observed",
        color=palette[g],
        xlabel="VOT (ms)",
        ylabel="Proportion /p/",
        title="Subject $(curdata.subject[1]) ($(lang))",
        legend=:topleft,
        background_color_legend = RGBA(0,0,0,.1),
        legendtitle = legtitle_txt, 
        legendtitlefontsize=8,
        legend_font_pointsize=8,
        linewidth=1.5,
        ylim=(0,1),
        xlim=(-20, 40)
    )

    𝓁 = mean(l); 𝒸 = mean(c); 𝒹 = mean(d); r = 𝒹 - 𝓁;
    hline!(fig, [𝓁], label=nothing, color=:gray63, linestyle=:dashdot, linewidth=1.0)
    hline!(fig, [𝒹], label=nothing, color=:gray63, linestyle=:dashdot, linewidth=1.0)
    plot!(fig, [𝒸, 𝒸], [0.0, 𝓁 + r/2], label=nothing, color=:grey63, linestyle=:dashdot, linewidth=1.0)
    plot!(fig, [-20.0, 𝒸], [𝓁 + r/2, 𝓁 + r/2], label=nothing, color=:grey63, linestyle=:dashdot, linewidth=1.0)
    plot!(fig, [x50, x50], [0, .5], label=nothing, color=:grey63, linestyle=:solid, linewidth=1.0)
    plot!(fig, [-20.0, x50], [.5, .5], label=nothing, color=:grey63, linestyle=:solid, linewidth=1.0)

    plot!(fig, stimContinuum, prob_p,
        ribbon = (prob_p - CIs_lo, CIs_hi - prob_p),
        fillalpha=.2,
        label="Predicted",
        color=palette[g],
        linestyle =:dash,
        linewidth=1.5
    )

    return fig
end

function plotFit_logreg_grp(chn, df, G)
    
    vals = get(chn, [:𝓁_sub, :𝒹_sub, :β_sub, :𝒸_sub])

    l = reduce(hcat, [vec(vals.𝓁_sub[i].data) for i in unique(df.S)])
    d = reduce(hcat, [vec(vals.𝒹_sub[i].data) for i in unique(df.S)])
    r = d - l
    b = reduce(hcat, [vec(vals.β_sub[i].data) for i in unique(df.S)])
    c = reduce(hcat, [vec(vals.𝒸_sub[i].data) for i in unique(df.S)])
    X = reshape(stimContinuum, 1, 1, :)
    prob_ps = l .+ (d .- l) .* logistic.(b .* (X .- c))
    means_subj = dropdims(mean(prob_ps; dims=1); dims=1)'
    means_grp = hcat([mean(means_subj[:, G .== g]; dims=2) for g in 1:3]...)

    hummeans = @chain df begin
        @group_by(G, lang_grp, VOT) 
        @summarize(Obs_P = sum(Obs_P), N = sum(N)) 
        @ungroup() 
        @mutate(mean_p = Obs_P ./ N)
        @arrange(G, VOT)
    end

    fig = plot(size=(420,360), left_margin=5Plots.mm);
    @df hummeans plot!(fig, :VOT, :mean_p;
        group=:lang_grp, 
        palette = [:red, :green, :blue],
        seriestype=:line,
        marker=:circle, lw=2, legend=:topleft,
        xlabel="VOT", ylabel="P(/p/)"
    );

    plot!(fig, stimContinuum, means_grp,
        label=nothing,
        palette = [:red, :green, :blue],
        seriestype=:line,
        linestyle =:dash,
        linewidth=1.5
    );

    return fig
end

# ================== END MODULE ==================
end