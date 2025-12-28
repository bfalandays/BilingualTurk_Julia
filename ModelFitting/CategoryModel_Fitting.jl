#= Note: NEW STRATEGY FOR USING DDM, STATE SPACE, AND MOUSE MODEL
    Fit the Category models for 2 and 4 categories. Then take the predictions to use as input
    for VOT function in the other models, and see if they predict RTs and mouse trajectories.

    Also try fitting the DDM and SSM in combination with the category model, but per-subject (not hierarchically)
=#

#using Distributed; (need = 9 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)
#@everywhere using BilingualTurk_Julia
using BilingualTurk_Julia.CategoryModel
using BilingualTurk_Julia.LogRegModel
# using OptimizationOptimJL: NelderMeadd
Random.seed!(1234)
# ======================================================================== 

# == PREPARING DATA == #
rawdf = DataFrame(CSV.File("../Exp2/Data/data.csv"))[:,:];
S, G, L, B, V, N, Y, df = prepare_data_cat(rawdf; subsample = false)

# == MCMC SAMPLER CONFIG == #
nchains = 6; 
niter = 10000; #10000;
nwarmup = 2000; #5000;
target_accept = .65;
# took ~4.8 hours with ReverseDiff compiled, niter=10000, nwarmup = 5000, target_accept=.8 (no subsampling)
# took 5.8 hrs with Mooncake, but the posteriors were much smoother and mixing was much better.

#= Testing on individual subs w/ non-hierarchical version
    s = df.S[df.subject .== 165][1]
    Vs = V[S .== s]
    Ns = N[S .== s]
    Ys = Y[S .== s]
    model = mod4cats(Vs, Ns, Ys)
    # model = mod2cats(Vs, Ns, Ys)
    chn = sample(Xoshiro(0), model, NUTS(nwarmup, target_accept; adtype=AutoMooncake()), MCMCThreads(), niter, nchains; progress=true, initial_params=fill(InitFromPrior(), nchains))
    chndf = DataFrame(summarize(chn))
    fig, elpd = plotFit(chn, df, s); display(fig)
=#

# == SAMPLING: 4 CAT MODEL == #
model_4 = mod4cats_hier(S, L, B, V, N, Y); # ppc = sample(model_4, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check

# map_estimate4 = maximum_a_posteriori(model_4, NelderMead(); adtype=AutoMooncake())

chn_4cats = sample(Xoshiro(0), model_4, NUTS(nwarmup, target_accept; max_depth=10, adtype=AutoMooncake()), MCMCThreads(), niter, nchains; progress=true, 
    # initial_params = fill(InitFromParams((logitw0=map_estimate4.values[:logitw0],)), nchains))
    # initial_params= fill(InitFromParams(map_estimate4), nchains))
    initial_params=fill(InitFromPrior(), nchains))
@save joinpath(@__DIR__, "Saved","mod4cats_hier_chains.jld2") chn_4cats
# @load joinpath(@__DIR__, "Saved","mod4cats_hier_chains.jld2") chn_4cats

describe(chn_4cats; sections=:internals)
chndf4, chndf4_sub, chndf4_grp = getChnDFs(chn_4cats, df)
notConv_4 = chndf4[chndf4.rhat .> 1.01,:]

chn = group(chn_4cats, "w0"); plot(chn)
chn = group(chn_4cats, "w_grp"); plot(chn)
chn = group(chn_4cats, "βb_w"); plot(chn)
chn = group(chn_4cats, "βl_w"); plot(chn)


# == SAMPLING: 2 CAT MODEL == #
model_2 = mod2cats_hier(S, L, B, V, N, Y); # ppc = sample(model_4, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check

# map_estimate2 = maximum_a_posteriori(model_2, NelderMead(); adtype=AutoMooncake())

chn_2cats = sample(Xoshiro(0), model_2, NUTS(nwarmup, target_accept; adtype=AutoMooncake()), MCMCThreads(), niter, nchains; progress=true, 
    # initial_params= fill(InitFromParams(map_estimate2), nchains))
    initial_params=fill(InitFromPrior(), nchains))
@save joinpath(@__DIR__, "Saved","mod2cats_hier_chains.jld2") chn_2cats
# @load joinpath(@__DIR__, "Saved","mod2cats_hier_chains.jld2") chn_2cats

describe(chn_2cats; sections=:internals)
chndf2, chndf2_sub, chndf2_grp = getChnDFs(chn_2cats, df)
notConv_2 = chndf2[chndf2.rhat .> 1.01,:]

chn = group(chn_2cats, "βb_μ"); plot(chn)
chn = group(chn_2cats, "βl_μ"); plot(chn)
# chn = group(chn_2cats, "βb_σ"); plot(chn)
# chn = group(chn_2cats, "βl_σ"); plot(chn)

# == SAMPLING: 4PL == #
S, G, V, N, Y, df = prepare_data_logreg(rawdf; subsample = false)

model_logreg = logreg_hier(S, G, V, N, Y); # ppc = sample(model_logreg, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check
chn_logreg = sample(Xoshiro(0), model_logreg, NUTS(nwarmup,target_accept; 
    adtype=AutoMooncake()), 
    # adtype=AutoReverseDiff(;compile=true)),
    MCMCThreads(), niter, nchains; progress=true, initial_params=fill(InitFromPrior(), 4))
@save joinpath(@__DIR__, "Saved","logreg_chains.jld2") chn_logreg
chndf_logreg = DataFrame(summarize(chn_logreg))

# == PLOTTING == #
# for s in sampled_subjects #[3,5,10,14,52] #
for (i, s) in enumerate(unique(df.subject))
    println("Plotting subject $i of $(length(unique(df.subject))); (subj = $s)")
    lang = df.lang_grp[df.subject .== s][1];

    # i = df.S[df.subject .== s][1]
    p1, elpd = plotFit(chn_4cats, df, i); # display(p1) 
    # fig = plot(p1, size = (400, 650))
    
    p2, elpd = plotFit(chn_2cats, df, i); # display(p2)

    # p3 = plotFit_logreg_subj(chn_logreg, df, i); # display(fig) 
    
    # layout = @layout([a b grid(2,1)])
    # fig = plot(p1, p2, p3, layout = layout, size = (1200, 800)); # display(fig)

    layout = grid(1,2)
    fig = plot(p1, p2, p3, layout = layout, size = (400, 800)); # display(fig)
    savefig(fig, "~/Documents/Projects/Bilingual_Turk/Exp2/Plots/Subj_plots/BayesianCategoryFitting/Hierarchical/preds/lang_$(lang)_subj$(s).pdf")
end

# ---- Post-hoc analysis ---- #

chn = chn_4cats#ppc
# region πₑ_μ_grp
    # plotting chains/posterior estimates for πₑ_μ_grp
    chn_w_μ_grp = group(chn, :w_grp)
    plot(chn_w_μ_grp)

    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]
    me_mnwts = [mean(chn, "w_sub[$i]") for i in findall(==(1), g_subj)]
    be_mnwts = [mean(chn, "w_sub[$i]") for i in findall(==(2), g_subj)]
    bs_mnwts = [mean(chn, "w_sub[$i]") for i in findall(==(3), g_subj)]
    density(me_mnwts, alpha=.3, xlims=(0,1), fill = (0, :blue));
    density!(be_mnwts, alpha=.3, fill = (0, :green));
    density!(bs_mnwts, alpha=.3, fill = (0, :red))

    # Check for differences between groups
    Δ₁₂ = chn["w_grp[1]"] - chn["w_grp[2]"]
    quantile(Δ₁₂, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₁₂)               # mean difference
    mean(Δ₁₂ .> 0)

    Δ₂₃ = chn["w_grp[2]"] - chn["w_grp[3]"]
    quantile(Δ₂₃, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₂₃)               # mean difference
    mean(Δ₂₃ .> 0)

    Δ₁₃ = chn["w_grp[1]"] - chn["w_grp[3]"]
    quantile(Δ₁₃, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₁₃)               # mean difference
    mean(Δ₁₃ .> 0)
# end

# region category means
    chn_bₑ_μ_grp = group(chn, :bₑ_μ_grp)
    plot(chn_bₑ_μ_grp)
    chn_pₑ_μ_grp = group(chn, :pₑ_μ_grp)
    plot(chn_pₑ_μ_grp)
    chn_bₛ_μ_grp = group(chn, :bₛ_μ_grp)
    plot(chn_bₛ_μ_grp)
    chn_pₛ_μ_grp = group(chn, :pₛ_μ_grp)
    plot(chn_pₛ_μ_grp)
# end

# region category SDs
    chn_σ = group(chn, :σ_grp)
    plot(chn_σ)

    chn_bₑ_σ_grp = group(chn, :bₑ_σ_grp)
    plot(chn_bₑ_σ_grp)
    chn_pₑ_σ_grp = group(chn, :pₑ_σ_grp)
    plot(chn_pₑ_σ_grp)
    chn_bₛ_σ_grp = group(chn, :bₛ_σ_grp)
    plot(chn_bₛ_σ_grp)
    chn_pₛ_σ_grp = group(chn, :pₛ_σ_grp)
    plot(chn_pₛ_σ_grp)
# end

## Remove workers at end
# rmprocs(workers()) 

############


S, G, V, N, Y, df = prepare_data_cat(@chain rawdf @filter(subject .== 3); subsample = false)
mod = mod4cats(V, N, Y)
chn = sample(Xoshiro(0), mod, NUTS(500,0.8; adtype=AutoMooncake()), MCMCThreads(), 1000, 4; progress=true, initial_params=fill(InitFromPrior(), 4))
chndf = DataFrame(summarize(chn))




