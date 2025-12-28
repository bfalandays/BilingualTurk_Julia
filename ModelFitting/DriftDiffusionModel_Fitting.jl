using BilingualTurk_Julia.DriftDiffusionModel
# ======================================================================== 

# == PREPARING DATA == #
rawdf = DataFrame(CSV.File("../Exp2/Data/data_BP.csv"));
# rawdf = @chain rawdf @filter(VOT .== maximum(VOT) || VOT .== minimum(VOT))
S, G, Vidx, V, P, R, df, Vstats = prepare_data_DDM(rawdf; subsample = true)

# == MCMC SAMPLER CONFIG == #
nchains = 4; 
niter = 1000;
nwarmup = 500;
target_accept = .65;

# == SAMPLING: 4 CAT MODEL == #
# mod(;track=false) = DDMhier(S, G, Vidx, P, R; track); # ppc = sample(model, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check
mod(;track=false) = catDDM(S, G, Vidx, P, R; track); # ppc = sample(model, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check
model = mod(); trackedmodel = mod(; track=true);
# varinfo = VarInfo(model); initvals = Dict(k => varinfo[k] for k in keys(varinfo))

# initvals = Dict(
#     @varname(v_grp)     => fill(0.0, (length(unique(V)), length(unique(G)))),
#     #@varname(σv)        => 2.0,
#     # @varname(t0)        => 0.001,
#     @varname(logitτ0)   => -7.0,
#     @varname(σt)        => 0.2,
#     @varname(a0)        => 1.0,
#     @varname(σa)        => 0.1,
#     @varname(z0)        => 0.5,
#     #@varname(σlogitz)   => 0.1,
# )
# initial_params = fill(InitFromParams(initvals), nchains);

initial_params= fill(InitFromPrior(), nchains) # initial_params=[rand(Xoshiro(i+1),Vector, model) for i in 1:nchains];

chn = sample(Xoshiro(0), model, NUTS(nwarmup, target_accept; adtype=AutoMooncake()), MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params)
# original model took ~5 hours (21240.1s) with niter=500, nwarmup=100, target_accept=.65, subsample=true (10 subs per group)
# reparameterized version took ~14 min (872.44s) with same settings -- 24x speedup!
# full dataset has 173 participants, so estimated time is 872s * (173/30) = ~5029s = ~84 min; actually took 14914s (~4.14 hours)

# @save joinpath(@__DIR__, "Saved","DDM_chains4.jld2") chn
# @load joinpath(@__DIR__, "Saved","DDM_chains4.jld2") chn

chndf = DataFrame(summarize(chn))
grpdf = summarize(chn[reduce(vcat, [
    namesingroup(chn, :v_grp), 
    namesingroup(chn, :a_grp),
    namesingroup(chn, :logitτ_grp)])])
grpdf = DataFrame(grpdf)

samples = Turing.returned(trackedmodel, chn);
rtplot, choiceplot = posterior_predictive_plot(samples, df);
display(rtplot); display(choiceplot);
savefig(rtplot, joinpath(@__DIR__, "Saved","DDMrtplot.pdf")); savefig(choiceplot, joinpath(@__DIR__, "Saved","DDMchoiceplot.pdf")); 

# Extract V (VOT index) and G (group index) from parameter names like "name[ v, g ]";
transform!(grpdf,
    :parameters => ByRow(p -> begin
        s = String(p)
        m = match(r"\[(\d+)\s*,\s*(\d+)\]$", s)   # match [v, g]
        m === nothing ? missing : parse(Int, m.captures[1])
    end) => :V,
    :parameters => ByRow(p -> begin
        s = String(p)
        m2 = match(r"\[(\d+)\s*,\s*(\d+)\]$", s)  # prefer [v, g]
        if m2 !== nothing
            return parse(Int, m2.captures[2])
        end
        m1 = match(r"\[(\d+)\]$", s)                 # fallback [g]
        m1 === nothing ? missing : parse(Int, m1.captures[1])
    end) => :G)
grpdf = @chain grpdf @filter(!ismissing(V)) @select(mean, V, G) @mutate(V = Float64.(V), mean = mean)

sort!(grpdf, [:G, :V])

@df grpdf plot(:V, :mean;
    group=:G, color=:G, seriestype=:line,
    marker=:circle, lw=2, legend=:topleft,
    xlabel="V", ylabel="mean", title="Mean vs V by Group")

chn_v_grp = MCMCChains.group(chn, :v_grp)
plot(chn_v_grp)

#######

model = catDDM(S, G, Vidx, V, P, R); #ppc = sample(model, Prior(), 5); c = DataFrame(summarize(ppc))
initial_params= fill(InitFromPrior(), nchains) # initial_params=[rand(Xoshiro(i+1),Vector, model) for i in 1:nchains];

chn = sample(Xoshiro(0), model, NUTS(nwarmup, target_accept; adtype=AutoMooncake()), MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params)

chndf = DataFrame(summarize(chn))

#################

# using ADTypes
# using DynamicPPL.TestUtils.AD: run_ad, ADResult
# using ForwardDiff, ReverseDiff, Mooncake

# resultF = run_ad(model, AutoForwardDiff(); benchmark=true); 
# @show resultF.grad_time / resultF.primal_time
# ERRORED

# resultR = run_ad(model, AutoReverseDiff(); benchmark=true);
# @show resultR.grad_time / resultR.primal_time
# #39.91

# resultM = run_ad(model, AutoMooncake(); benchmark=true);
# @show resultM.grad_time / resultM.primal_time
# # 21.39
