using Distributed; (need = 8 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)

@everywhere using BilingualTurk_Julia

#= Data pre-processing -- saved result to not repeat
    rawdata = DataFrame(CSV.File("../Exp2/Data/data.csv"))[:,2:end];
    data = @chain rawdata @select(mt_id, subject, trial, language, votstep, VOT, choseP, MAD, MD_above, AD, RT, AUC)
    data.vot_norm = (data.VOT + 20)/(60) * 2 - 1 #standardize(ZScoreTransform, data.VOT)
    data = subject_to_idx(data);
    data = @chain data @mutate(G = case_when(language == "Monolingual English" => 1,language == "Bilingual English" => 2,language == "Bilingual Spanish" => 3))

    mtdata = DataFrame(CSV.File("../Exp2/Data/mt_data_long.csv"))[:,:];
    data.sampEn .= 0.0; data.MD₂ .= 0.0; data.AD₂ .= 0.0; 
    for (i, mtid) in enumerate(unique(data.mt_id))
        progress = i/length(unique(data.mt_id))
        println(round(progress, digits=2))

        x = mtdata.xpos[mtdata.mt_id .== mtid]
        y = mtdata.ypos[mtdata.mt_id .== mtid]
        MD, AD = devMeasures(collect(zip(x,y)))

        ẋ = diff(x)
        sampEn = sampleEntropy(ẋ, 3)
        data[data.mt_id .== mtid, [:MD₂, :AD₂, :sampEn]] .= [MD AD sampEn]
    end
    @save "../Exp2/Data/dataM.jld2" data
=#
@load "../Exp2/Data/dataM.jld2" data

subsample = true
if subsample 
    println("Subsampling data for quicker testing")
    Random.seed!(1)
    sampled_subjects = @chain data begin
        @select(subject, language)
        @distinct()
        @group_by(language)
        @slice_sample(n = 2, replace=false)
        @ungroup()
        @arrange(subject)
        @pull(subject)
    end
    data = @chain data @filter(subject in !!sampled_subjects)
    data = subject_to_idx(data)
end

S = data.S;
G = data.G;
V = data.vot_norm;
y = data.AD₂;

# μₑ, σₑ, k, cₖ = [0.3, 0.2, 59.75, 0.61]; m=3; r=0.0;

# ======================================================================== 

# Parameters for the MCMC sampling
nchains = 4;
niter = 2000;
nwarmup = 2000;

model = mDDM(S, G, V, y)
# ppc = sample(model, Prior(), 100); # c = DataFrame(summarize(ppc)) # prior predictive check
initial_params=[rand(Xoshiro(i+1),Vector, model) for i in 1:nchains];
chn_ssmod = sample(model, MH(), MCMCDistributed(), niter, nchains; progress=true, initial_params=initial_params)
chndf = DataFrame(summarize(chn_ssmod))
chndict = Dict(chndf.parameters .=> chndf.mean)
notConv = chndf[chndf.rhat .> 1.01,:]

# ---------------------------------- #


