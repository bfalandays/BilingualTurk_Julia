using Distributed; (need = 8 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)

@everywhere using BilingualTurk_Julia

rawdata = DataFrame(CSV.File("../Exp2(lab)_forPub/Data/data.csv"))[:,:];
summaryData = combine(groupby(rawdata, [:subject, :votstep, :language]), :choseP => sum => :Obs_P, nrow => :N)
data = subject_to_idx(summaryData)
df9 = rawdata[rawdata.votstep .== 9, :];
# mtdata = DataFrame(CSV.File("../Exp2(lab)_forPub/Data/mtdata_preprocd_long.csv"))[:,2:12];

# ========================================================================

g(params) = optimFunc(df9, params)

# params are: σ, σ_τ, k, c_scalar
initParams = [.5, .1, 5.0, 1.0]
lower_ = [.1, 1e-6, .1, .5]
upper_ = [1.0, .5, 100.0, 2.0]

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
# converged with: 
# 0.698
# 0.449
# 59.75
# 0.61

σ, σ_τ, k, c_scalar = bestFitparams

simMDs = pmap((trial) -> simTrial_getMD(.9, σ,σ_τ, k, c_scalar), 1:length(df9.MD_above));
density(simMDs, xlims=(0,2))
density!(rawdata.MD_above[rawdata.votstep .== 5], xlims=(0,2))

means = []
i =0
for A in .5:.1:1.0    
    i += 1
    println(round(i/101, digits=2))
    simMDs = pmap((trial) -> simTrial_getMD(A, σ,σ_τ, k, c_scalar), 1:length(df9.MD_above));
    push!(means, mean(simMDs))
end