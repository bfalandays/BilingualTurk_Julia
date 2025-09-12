using Distributed; (need = 8 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)

@everywhere using BilingualTurk_Julia

@load "../Exp2/Data/dataM.jld2" data
df9 = data[data.votstep .== 9, :];

# ========================================================================

g(params) = optimFunc(df9, params)

# params are: μₑ, σₑ, k, cₖ, Aₖ
initParams = [.5, .1, 5.0, 1.0, 1.0]
lower_ = [.1, 1e-6, .1, .5, .1]
upper_ = [1.0, .5, 100.0, 2.0, 10.0]

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
# 0.6565015230399541
# 0.42850736749131746
# 80.87909538460718
# 1.0916281635898868
# 1.9434464895468524

μₑ, σₑ, k, cₖ, Aₖ = bestFitparams

#sim_ys = pmap((trial) -> simTrial_getMD(.9, σ,σ_τ, k, c_scalar), 1:length(df9.MD_above));
sim_ys = pmap((trial) -> simTrial_getMeasures(1.0*Aₖ; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, out="MD"), 1:5000);

using StatsPlots
density(data.MD₂, xlims=(0,2))
density!(sim_ys, xlims=(0,2))

means = []
i =0
for A in .5:.1:1.0    
    i += 1
    println(round(i/101, digits=2))
    sim_ys = pmap((trial) -> simTrial_getMeasures(1.0*Aₖ; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, out="AD"), 1:5000);
    push!(means, mean(sim_ys))
end

plot(.5:.1:1.0, means)