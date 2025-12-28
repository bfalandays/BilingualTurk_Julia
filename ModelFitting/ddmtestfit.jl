using BilingualTurk_Julia.DriftDiffusionModel
# using Mooncake
# ======================================================================== 

# == PREPARING DATA == #
df = DataFrame(CSV.File("../Exp2/Data/CavanaghThetaNN.csv"));

df.subj_idx = df.subj_idx .+ 1
df = @chain df @mutate(C = case_when(conf == "LC" => 1, conf == "HC" => 2))
df.response = Int64.(df.response)

S = df.subj_idx;
C = df.C;
P = df.response;
R = df.rt;

# == MCMC SAMPLER CONFIG == #
nchains = 4;
niter = 50;
nwarmup = 25;
target_accept = .65;

# == SAMPLING: 4 CAT MODEL == #
model = DDMhierTest(S, C, P, R); # ppc = sample(model, Prior(), 100); c = DataFrame(summarize(ppc)) # prior predictive check

varinfo = VarInfo(model); initvals = Dict(k => varinfo[k] for k in keys(varinfo))
initvals = Dict(
    @varname(v_grp)     => [2.0, 2.0],
    #@varname(σv)        => 2.0,
    @varname(t0)        => 0.001,
    @varname(σt)        => 0.2,
    @varname(a0)        => 1.0,
    @varname(σa)        => 0.1,
    @varname(z0)        => 0.5,
    #@varname(σlogitz)   => 0.1,
)
# retval, varinfo_new = DynamicPPL.init!!(Xoshiro(1), model, VarInfo(model), InitFromParams(initvals))
# initvals = Dict(k => varinfo_new[k] for k in keys(varinfo_new))

initial_params = fill(InitFromParams(initvals), nchains);
# initial_params = fill(InitFromPrior(), nchains);

# chn = sample(Xoshiro(0), model, NUTS(nwarmup, target_accept;adtype=AutoReverseDiff(;compile=false)), MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params)
chn = sample(Xoshiro(0), model, NUTS(nwarmup, target_accept;adtype=AutoForwardDiff()), MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params)

chndf = DataFrame(summarize(chn))
chn_v_grp = MCMCChains.group(chn, :v_grp)
plot(chn_v_grp)

@load joinpath(@__DIR__, "Saved","DDMtestchn1.jld2") chn

using ADTypes
using DynamicPPL.TestUtils.AD: run_ad, ADResult
using ForwardDiff, ReverseDiff, Mooncake

for adtype in [AutoForwardDiff(), AutoReverseDiff(), AutoMooncake(), AutoMooncakeForward()]
    result = run_ad(model, adtype; benchmark=true)
    @show result.grad_time / result.primal_time
end

for adtype in [AutoZygote()]
    result = run_ad(model, adtype; benchmark=true)
    @show result.grad_time / result.primal_time
end