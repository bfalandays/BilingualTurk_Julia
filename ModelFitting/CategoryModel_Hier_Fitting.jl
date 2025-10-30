#using Distributed; (need = 9 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)
using StatsPlots
#@everywhere using BilingualTurk_Julia
using BilingualTurk_Julia
# ======================================================================== 

# == PREPARING DATA == #
df = DataFrame(CSV.File("../Exp2/Data/data.csv"))[:,:];
S, G, V, N, Y, df, Vstats = prepare_data_cat(df; subsample = true)

# == MCMC SAMPLER CONFIG == #
nchains = 4;
niter = 1000;
nwarmup = 500;

# == SAMPLING: 4 CAT MODEL == #
model_4 = mod4cats_hier(S, G, V, N, Y); # ppc = sample(model_4, Prior(), 100); # c = DataFrame(summarize(ppc)) # prior predictive check
# histogram(ppc["bₑ_μ_sub[1]"]); histogram(ppc["πₑ_sub[1]"]); histogram(ppc["bₑ_σ_sub[1]"])
initial_params=[rand(Xoshiro(i+1),Vector, model_4) for i in 1:nchains];
chn_4cats = sample(Xoshiro(0), model_4, NUTS(nwarmup, .65; adtype=AutoReverseDiff(;compile=true)), MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params)

chndf4_sub = getChnDFs4plot(chn_4cats, df, Vstats)

chndf4_grp = @chain chndf4 @filter(!occursin(r"(sub)", String(parameters))) @filter(occursin(r"\[(\d+)\]$", String(parameters)))

    # 
    # 
    # @filter(!occursin(r"(log)", String(parameters)))
    # 
    # @filter(!occursin(r"\[\s*-?\d+\s*,\s*-?\d+\s*\]$", String(parameters)))  # drop names ending with [A,B]

transform!(chndf_4_sub, 
    :parameters => ByRow(p -> begin
        s = String(p)                           # convert Symbol → String
        m = match(r"\[(\d+)\]$", s)             # find trailing [N]
        m === nothing ? missing : parse(Int, m.captures[1])
    end) => :S)
transform!(chndf_4_sub,
    :parameters => ByRow(p -> replace(String(p), r"\[\d+\]$" => "")) => :param)

df4_sub_wide.G = G

notConv_4 = chndf_4[chndf_4.rhat .> 1.01,:]
@save joinpath(@__DIR__, "Saved","mod4cats_hier_chains.jld2") chn_4cats
# chn_4cats = load(joinpath(@__DIR__, "Saved","mod4cats_hier_chains.jld2"), "chn_4cats")

# == SAMPLING: 2 CAT MODEL == #
model_2 = mod2cats_hier(S, G, V, N, y)
# ppc = sample(model_2, Prior(), 1000) # prior predictive check
# histogram(ppc["bₑ_σ_sub[1]"])
initial_params=[rand(Xoshiro(i),Vector, model_2) for i in 1:nchains];
chn_2cats = sample(Xoshiro(0), model_2, NUTS(; adtype=AutoReverseDiff(;compile=true)), MCMCDistributed(), niter, nchains; nadapts=nwarmup, progress=true, initial_params=initial_params)
chndf_2 = DataFrame(summarize(chn_2cats))
chndf_2 = @chain chndf_2 begin
    @filter(!occursin(r"(logit)", String(parameters)))
    @filter(!occursin(r"(_z)", String(parameters)))
    @filter(!occursin(r"\[\s*-?\d+\s*,\s*-?\d+\s*\]$", String(parameters)))  # drop names ending with [A,B]
end
notConv_2 = chndf_2[chndf_2.rhat .> 1.01,:]
@save joinpath(@__DIR__, "Saved","mod2cats_hier_chains.jld2") chn_2cats
# chn_2cats = load(joinpath(@__DIR__, "Saved","mod2cats_hier_chains.jld2"), "chn_2cats")

# == PLOTTING == #
i=0
# for subj in sampled_subjects #[3,5,10,14,52] #
for subj in unique(data.subject)
    
    i += 1; println("Plotting subject $i of $(length(unique(data.subject))); (subj = $subj)")
    lang = data.language[data.subject .== subj][1];

    p1, elpd = plotFit(chn_4cats, data, subj); #display(p1)
    fig = plot(p1, size = (400, 650))
    
    # p2, elpd = plotFit(chn_2cats, data, subj); #display(p2)
    
    # fig = plot(p1, p2, layout = (1, 2), size = (900, 600)); #display(fig)
    savefig(fig, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/Hierarchical/preds/pred_$(lang_map[lang])_subj$(subj).pdf")

end

# ---- Post-hoc analysis ---- #

chn = chn_4cats#ppc
# region πₑ_μ_grp
    # plotting chains/posterior estimates for πₑ_μ_grp
    chn_πₑ_μ_grp = group(chn, :πₑ_grp)
    plot(chn_πₑ_μ_grp)

    g_subj = [group_map[data.language[data.subject .== s][1]] for s in unique(data.subject)]
    me_mnwts = [mean(chn, "πₑ_sub[$i]") for i in findall(==(1), g_subj)]
    be_mnwts = [mean(chn, "πₑ_sub[$i]") for i in findall(==(2), g_subj)]
    bs_mnwts = [mean(chn, "πₑ_sub[$i]") for i in findall(==(3), g_subj)]
    density(me_mnwts, alpha=.3, xlims=(0,1), fill = (0, :blue));
    density!(be_mnwts, alpha=.3, fill = (0, :green));
    density!(bs_mnwts, alpha=.3, fill = (0, :red))

    # Check for differences between groups
    Δ₁₂ = chn["πₑ_grp[1]"] - chn["πₑ_grp[2]"]
    quantile(Δ₁₂, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₁₂)               # mean difference
    mean(Δ₁₂ .> 0)

    Δ₂₃ = chn["πₑ_grp[2]"] - chn["πₑ_grp[3]"]
    quantile(Δ₂₃, [0.025, 0.975])     # 95% credible interval for the difference
    mean(Δ₂₃)               # mean difference
    mean(Δ₂₃ .> 0)

    Δ₁₃ = chn["πₑ_grp[1]"] - chn["πₑ_grp[3]"]
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



