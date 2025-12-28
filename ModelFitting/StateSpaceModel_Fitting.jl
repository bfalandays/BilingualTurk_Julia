#using Distributed; (need = 5 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)
# @everywhere using BilingualTurk_Julia
using BilingualTurk_Julia.StateSpaceModel
# ======================================================================== 

# == PREPARING DATA == #
df = DataFrame(CSV.File("../Exp2/Data/mt_data_long_BP.csv"))[:,:];
S, G, V, Y, D, Κ, Am, df = prepare_data_SS(df; subsample = true)
R = (1 ./ sqrt.(Κ));

# == INITIALIZING MODEL == #
model = ssmod(S, G, V, Y, R, Am); # ppc = sample(model, Prior(), 100); ppcdf = DataFrame(summarize(ppc))

# == MCMC SAMPLER CONFIG == #
nchains = 4; 
niter = 250#500;
nwarmup = 50# 250;
target_accept = .65;

initial_params= fill(InitFromPrior(), nchains) # initial_params=[rand(Xoshiro(i+1),Vector, model) for i in 1:nchains];

# == SAMPLING == #
chn = sample(model, 
    NUTS(nwarmup, target_accept; adtype=AutoReverseDiff(;compile=true)), #adtype=AutoMooncake()), 
    MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params);
# chn = sample(model, HMC(.05,10), MCMCThreads(), niter, nchains; progress=true, initial_params=initial_params); # with HMC, takes 665 second for 500 samples

# == PLOTTING == #
chndf = DataFrame(summarize(chn)); # chndict = Dict(chndf.parameters .=> chndf.mean)
# notConv = chndf[chndf.rhat .> 1.01,:]

plot_zMAP_subj(chn, S, G, V, 1)
