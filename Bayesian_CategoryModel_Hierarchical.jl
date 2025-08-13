using Distributed
addprocs(8; exeflags="--project")

@everywhere include("/Users/jfalanda/Documents/Projects/Bilingual_Turk/BilingualTurk_Julia/src/BayesianModelFuncs.jl")

#= 1 chain (for testing)
    data = subject_to_idx(summaryData[summaryData.language .== "Monolingual English", :])
    niter = 1000 
    nwarmup = 1000 
    
    mod_4cats = mod4cats_hier(data)
    chn_4cats = sample(Xoshiro(0), mod_4cats, NUTS(; adtype=AutoReverseDiff(;compile=true)), niter; nadapts=nwarmup, progress=true, initial_params=rand(Xoshiro(1),Vector, mod_4cats))

    mod_2cats = mod2cats_hier(data)
    chn_2cats = sample(Xoshiro(0), mod_2cats, NUTS(; adtype=AutoReverseDiff(;compile=true)), niter; nadapts=nwarmup, progress=true, initial_params=rand(Xoshiro(1),Vector, mod_2cats))
=#

# Parameters for the MCMC sampling
nchains = 4
niter = 5000 
nwarmup = 5000 

data = subject_to_idx(summaryData)

# randomly sample 15 subjects from unique subjects
sampled_subjects = sample(Xoshiro(0), unique(data.subject), 15, replace=false)
data = data[in.(data.subject, Ref(sampled_subjects)), :]
data = subject_to_idx(data)

# count the number of subjects in each language
subject_counts = combine(groupby(data, :language), nrow => :count)

# ---- Fit the 4 category model ---- #
mod_4cats = mod4cats_hier(data)
initial_params=[rand(Xoshiro(i+1),Vector, mod_4cats) for i in 1:nchains]
chn_4cats = sample(Xoshiro(0), mod_4cats, NUTS(; adtype=AutoReverseDiff(;compile=true)), MCMCDistributed(), niter, nchains; nadapts=nwarmup, progress=true, initial_params=initial_params)
#@save "mod4cats_hier_chains.jld2" chn_4cats
# @load "mod4cats_hier_chains.jld2"

# ---- Fit the 2 category model ---- #
mod_2cats = mod2cats_hier(data)
initial_params=[rand(Xoshiro(i),Vector, mod_2cats) for i in 1:nchains]
chn_2cats = sample(Xoshiro(0), mod_2cats, NUTS(; adtype=AutoReverseDiff(;compile=true)), MCMCDistributed(), niter, nchains; nadapts=nwarmup, progress=true, initial_params=initial_params)
#@save "mod2cats_hier_chains.jld2" chn_2cats
# @load "mod2cats_hier_chains.jld2"

i=0
for subj in sampled_subjects #[3,5,10,14,52] #
#for subj in unique(data.subject)
    i += 1
    println("Plotting subject $i of $(length(unique(data.subject)))")
    lang = data.language[data.subject .== subj][1]

    p1, waic4cats = plotFit(chn_4cats, data, subj); display(p1)
    #p2, waic2cats = plotFit(chn_2cats, data, subj); #p2
    #fig = plot(p1, p2, layout = (1, 2), size = (900, 600)); #fig 
    # display(fig)
    
    #savefig(fig, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/Hierarchical/preds/pred_$(lang_map[lang])_subj$(subj).pdf")

end

# region check rhat for all parameters
    c = DataFrame(summarize(chn_4cats))
    notConv = c[c.rhat .> 1.01,:]#, rhat =c.rhat[c.rhat .> 1.01], mean = c.mean[c.rhat .> 1.01])
# end

chn = chn_2cats
# region πₑ_μ_grp
    # plotting chains/posterior estimates for πₑ_μ_grp
    chn_πₑ_μ_grp = group(chn, :πₑ_μ_grp)
    plot(chn_πₑ_μ_grp)

    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]
    me_mnwts = [mean(chn, "πₑ_μ_sub[$i]") for i in findall(==(1), g_subj)]
    be_mnwts = [mean(chn, "πₑ_μ_sub[$i]") for i in findall(==(2), g_subj)]
    bs_mnwts = [mean(chn, "πₑ_μ_sub[$i]") for i in findall(==(3), g_subj)]
    density(me_mnwts, alpha=.3, xlims=(.6,1), fill = (0, :blue));
    density!(be_mnwts, alpha=.3, fill = (0, :green));
    density!(bs_mnwts, alpha=.3, fill = (0, :red))

    # Check for differences between groups
    Δ₁₂ = chn["πₑ_μ_grp[1]"] - chn["πₑ_μ_grp[2]"]
    quantile(Δ₁₂, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₁₂)               # mean difference
    mean(Δ₁₂ .> 0)

    Δ₂₃ = chn["πₑ_μ_grp[2]"] - chn["πₑ_μ_grp[3]"]
    quantile(Δ₂₃, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₂₃)               # mean difference
    mean(Δ₂₃ .> 0)

    Δ₁₃ = chn["πₑ_μ_grp[1]"] - chn["πₑ_μ_grp[3]"]
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
rmprocs(workers()) 



