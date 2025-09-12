using Distributed; (need = 9 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)
using StatsPlots
@everywhere using BilingualTurk_Julia

mtdata = DataFrame(CSV.File("../Exp2/Data/mt_data_long.csv"))[:,:];

subsample = false
if subsample 
    println("Subsampling data for quicker testing")
    Random.seed!(1)
    sampled_subjects = @chain mtdata begin
        @select(subject, language)
        @distinct()
        @group_by(language)
        @slice_sample(n = 10, replace=false)
        @ungroup()
        @arrange(subject)
        @pull(subject)
    end
    mtdata = @chain mtdata @filter(subject in !!sampled_subjects)
    mtdata = subject_to_idx(mtdata)
end

#mtdata.vot_norm = standardize(ZScoreTransform, mtdata.VOT)
mtdata = @chain mtdata begin
    @mutate(ang = atan.(ypos, xpos), vot_norm = (VOT + 20)/(60) * 2 - 1)
    @mutate(G = case_when(language == "Monolingual English" => 1,
                          language == "Bilingual English" => 2,
                          language == "Bilingual Spanish" => 3))
    @select(subject, S, trial, G, votstep, vot_norm, ang, mt_seq)
    @pivot_wider(names_from = mt_seq, values_from = ang)
end
S = mtdata.S
G = mtdata.G
V = mtdata.vot_norm
y =  Matrix{Float64}(@chain mtdata @select(Symbol("1"):Symbol("101")))

# ======================================================================== 

# Parameters for the MCMC sampling
nchains = 4;
niter = 2000;
nwarmup = 2000;

model = ssmod(S, G, V, y)
# ppc = sample(model, Prior(), 100);# c = DataFrame(summarize(ppc)) # prior predictive check
initial_params=[rand(Xoshiro(i+1),Vector, model) for i in 1:nchains];
chn_ssmod = sample(model, MH(), MCMCDistributed(), niter, nchains; progress=true, initial_params=initial_params)
chndf = DataFrame(summarize(chn_ssmod))
chndict = Dict(chndf.parameters .=> chndf.mean)
notConv = chndf[chndf.rhat .> 1.01,:]

#### 

zₛₙ, πlogistic = getMAP(chndict, mtdata)
for s in unique(mtdata.S)
    # s = 1
    p = plotSubjMAP(zₛₙ, πlogistic, mtdata, s)
    display(p)
end