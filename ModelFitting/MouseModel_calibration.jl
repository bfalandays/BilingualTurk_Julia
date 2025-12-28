using Distributed; (need = 9 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)
@everywhere using BilingualTurk_Julia.MouseModel
# using BilingualTurk_Julia.MouseModel

#= Data pre-processing -- saved result to not repeat
    rawdata = DataFrame(CSV.File("../Exp2/Data/data_BP.csv"))[:,2:end];
    data = @chain rawdata @select(mt_id, subject, trial, lang_grp, votstep, VOT, choseP, MAD, MD_above, AD, RT, AUC)
    data = subject_to_idx(data);
    data = @chain data @mutate(G = case_when(lang_grp == "BE" => 1,lang_grp== "BS" => 2,lang_grp == "ME" => 3))

    mtdata = DataFrame(CSV.File("../Exp2/Data/mt_data_long_BP.csv"))[:,:];
    data.sampEn .= 0.0; data.MD₂ .= 0.0; data.AD₂ .= 0.0; 
    for (i, mtid) in enumerate(unique(data.mt_id))
        progress = i/length(unique(data.mt_id))
        println(round(progress, digits=2))

        x = mtdata.xpos[mtdata.mt_id .== mtid]
        y = mtdata.ypos[mtdata.mt_id .== mtid]
        MD, AD = getDevMeasures(x,y)

        ẋ = diff(x)
        sampEn = sampleEntropy(ẋ, 3)
        data[data.mt_id .== mtid, [:MD₂, :AD₂, :sampEn]] .= [MD AD sampEn]
    end
    @save "../Exp2/Data/dataM.jld2" data
=#

# data = DataFrame(CSV.File("../Exp2/Data/data_BP.csv"))[:,:];
@load "../Exp2/Data/dataM.jld2" data
df9 = data[data.votstep .== 9 .&& data.lang_grp .== "ME", :];

# ========================================================================

g(params) = optimFunc(df9, params)

# params are: β, k, cₖ
initParams = [5.0, 250.0, 1.0]
lower_ = [.1, 1.0, .1]
upper_ = [10.0, 500.0, 2.0]

#res = optimize(g, lower_, upper_, initParams, Fminbox(BFGS()));
#res = optimize(g, initParams, BFGS());
# res = optimize(g, lower_, upper_, initParams, Fminbox(NelderMead()), Optim.Options(iterations=100, outer_iterations=100))
res = optimize(g, lower_, upper_, initParams, Fminbox(NelderMead()),#
                Optim.Options( 
                    show_trace = true, #store_trace = true,
                    iterations=100, outer_iterations=100
                ))
convergence = Optim.converged(res)

bestFitparams = Optim.minimizer(res)
β, k, cₖ = bestFitparams
# β, k, cₖ = 3.45, 217.23, 1.86

simdata = pmap((trial) -> simTrial_getMeasures(1.0; β = β, k = k, cₖ = cₖ, out="MD"), 1:length(df9.MD_above));
sim_ys = first.(simdata); sim_choices = last.(simdata);
B_ys = sim_ys[sim_choices .== 0]; P_ys = sim_ys[sim_choices .== 1];

using StatsPlots
humdata = df9.MD₂
density(humdata,  xlims=(minimum(humdata), maximum(humdata)), alpha=.5, label="Human", bins=100);
density!(sim_ys, xlims=(minimum(humdata), maximum(humdata)), alpha=.5, label="Sim", bins=100)

# density!(P_ys, xlims=(minimum(humdata), maximum(humdata)))
# density!(B_ys, xlims=(minimum(humdata), maximum(humdata)))


means = []
i =0
for A in .5:.1:1.0    
    i += 1
    println(round(i/101, digits=2))
    sim_ys = pmap((trial) -> simTrial_getMeasures(1.0*Aₖ; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, out="AD"), 1:5000);
    push!(means, mean(sim_ys))
end

plot(.5:.1:1.0, means)