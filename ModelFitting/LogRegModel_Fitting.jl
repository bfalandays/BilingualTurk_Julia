using BilingualTurk_Julia.LogRegModel
# ======================================================================== 

# == PREPARING DATA == #
rawdf = DataFrame(CSV.File("../Exp2/Data/data.csv"))[:,:];
S, G, V, N, Y, df = prepare_data_logreg(rawdf; subsample = false)

# == MCMC SAMPLER CONFIG == #
nchains = 4; 
niter = 1000;
nwarmup = 500;
target_accept = .8;

# == SAMPLING: HIERARCHICAL LOGISTIC REGRESSION MODEL == #
model_logreg = logreg_hier(S, G, V, N, Y); # ppc = sample(model_logreg, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check
chn_logreg = sample(Xoshiro(0), model_logreg, NUTS(nwarmup,target_accept; 
    adtype=AutoMooncake()), 
    # adtype=AutoReverseDiff(;compile=true)),
    MCMCThreads(), niter, nchains; progress=true, initial_params=fill(InitFromPrior(), 4))
@save joinpath(@__DIR__, "Saved","logreg_chains.jld2") chn_logreg
chndf_logreg = DataFrame(summarize(chn_logreg))

# plotting subject-level fits
for (i, s) in enumerate(unique(S))
    subject = first(df.subject[df.S .== s])
    lang = df.lang_grp[df.S .== s][1];

    fig = plotFit_logreg_subj(chn_logreg, df, s); # display(fig) 
    savefig(fig, "~/Documents/Projects/Bilingual_Turk/Exp2/Plots/Subj_plots/BayesianCategoryFitting/LogReg/preds/lang_$(lang)_subj_$(subject).pdf")
end

fig = plotFit_logreg_grp(chn_logreg, df, G);
savefig(fig, "~/Documents/Projects/Bilingual_Turk/Exp2/Plots/LogReg_preds.pdf")

boundarydf = getBoundaryDF(chn_logreg, df)
plot(group(chn_logreg, :𝒸0))

# rawdf = leftjoin(rawdf, select(boundarydf, :subject, :boundary => :boundary2, :x50 => :x502), on = :subject)
# rawdf.deltaVOT_x50_jul = rawdf.VOT .- rawdf.x502
# CSV.write("../Exp2/Data/data2.csv", rawdf)

chn=chn_logreg
Δ₁₂ = chn["𝒸0[1]"] - chn["𝒸0[2]"];
quantile(Δ₁₂, [0.025, 0.975])     # 95% credible interval for the difference
mean(Δ₁₂)               # mean difference

Δ₂₃ = chn["𝒸0[2]"] - chn["𝒸0[3]"];
quantile(Δ₂₃, [0.025, 0.975])     # 95% credible interval for the difference
mean(Δ₂₃)               # mean difference

Δ₁₃ = chn["𝒸0[1]"] - chn["𝒸0[3]"];
quantile(Δ₁₃, [0.025, 0.975])     # 95% credible interval for the difference
mean(Δ₁₃)               # mean difference
# == #