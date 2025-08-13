using Distributed
addprocs(4; exeflags="--project")

@everywhere include("/Users/jfalanda/Documents/Projects/Bilingual_Turk/BilingualTurk_Julia/src/BayesianModelFuncs.jl")

# Fitting for each subject
subject_chains = Dict()
model_comparison = Dict()

# Parameters for the MCMC sampling
nchains = 4
niter = 5000
nwarmup = 5000 

count = 0
for subj in unique(summaryData.subject) #(14, 44) #
    count += 1
    println("Progress: ", round(count / length(unique(summaryData.subject)), digits=3), " ; Subject: ", subj)

    # subj = 11
    curdata = summaryData[summaryData.subject .== subj, :]
    lang = unique(curdata.language)[1]

    # ---- Fit the 4 category model ---- #
    println("\tFitting 4 cat model...")
    
    mod_4cats = mod4cats(curdata)
    initial_params=[rand(Xoshiro(i + 100),Vector, mod_4cats) for i in 1:nchains]
    chn_4cats = sample(Xoshiro(0), mod_4cats, NUTS(), MCMCDistributed(), niter, nchains; nadapts=nwarmup, progress=true, initial_params=initial_params)
    # summarize(chn_4cats)
    # plot(chn_4cats)
    p, _ = plotFit(chn_4cats, curdata, subj)
    display(p)
        
    # ---- Fit the 2 category model ---- #
    println("\tFitting 2 cat model...")

    mod_2cats = mod2cats(curdata)
    initial_params=[rand(Xoshiro(i + 100),Vector, mod_2cats) for i in 1:nchains]
    chn_2cats = sample(Xoshiro(0), mod_2cats, NUTS(), MCMCDistributed(), niter, nchains; nadapts=nwarmup, progress=true, initial_params=initial_params)

    # ---- Plotting ---- #
    println("\t\tPlotting...")

    pred_4cats_plt, waic_4cats = plotFit(chn_4cats, curdata, subj);
    pred_2cats_plt, waic_2cats = plotFit(chn_2cats, curdata, subj);
    combinedfig = plot(pred_4cats_plt, pred_2cats_plt, layout = (1,2), size=(1000,800))

    chn_2cats_plt = plot(chn_2cats);
    chn_4cats_plt = plot(chn_4cats);

    savefig(combinedfig, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/NonHierarchical/preds/pred_$(lang_map[lang])_subj$(subj).pdf");
    # savefig(pred_4cats_plt, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/NonHierarchical/preds/pred_$(lang_map[lang])_subj$(subj)_4cats.pdf");
    # savefig(pred_2cats_plt, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/preds/pred_$(lang_map[lang])_subj$(subj)_2cats.pdf");
    savefig(chn_4cats_plt, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/NonHierarchical/chains/chn_$(lang_map[lang])_subj$(subj)_4cats.pdf");
    savefig(chn_2cats_plt, "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Plots/Subj_plots/BayesianCategoryFitting/NonHierarchical/chains/chn_$(lang_map[lang])_subj$(subj)_2cats.pdf");

    # ---- Storing Data ---- #
    subject_chains[subj] = (chn_4cats=chn_4cats, chn_2cats=chn_2cats)
    model_comparison[subj] = (lang = unique(curdata.language), EngWt = mean(chn_4cats[:π_μ]), WAIC_4cats=waic_4cats, WAIC_2cats=waic_2cats, ΔWAIC = waic_2cats - waic_4cats, bestfit = ifelse(waic_2cats < waic_4cats, "2cats", "4cats"))

end
#@save "~/Documents/Projects/Bilingual_Turk/Exp2(lab)_forPub/Data/subject_chains.jld2" subject_chains
@save "subject_chains.jld2" subject_chains

## Remove workers at end
rmprocs(workers())

df = DataFrame(subject=keys(subject_chains), EngWt=getindex.(values(subject_chains), :EngWt), lang=getindex.(values(subject_chains), :lang))
df.SpnWt = 1 .- df.EngWt; df.EngWt_jitter = df.EngWt .+ 0.02 .* randn(nrow(df)); df.SpnWt_jitter = df.SpnWt .+ 0.02 .* randn(nrow(df))
scatter(df.EngWt_jitter, df.SpnWt_jitter; group=df.lang, alpha=0.1, color=[group_colors[l] for l in df.lang], xlabel="English weight (πₑ)", ylabel="Spanish weight (1 - πₑ)", xlim=(0,1), ylim=(0,1), size=(400,400), label="")
for (g, c) in group_colors scatter!([NaN], [NaN]; label=g, markercolor=c) end
group_means = combine(groupby(df, :lang), :EngWt => mean => :EngWt_mean, :SpnWt => mean => :SpnWt_mean)
scatter!(group_means.EngWt_mean, group_means.SpnWt_mean; markercolor=[group_colors[l] for l in group_means.lang], markersize=12, markerstrokecolor=:black, alpha=1.0, label=false)