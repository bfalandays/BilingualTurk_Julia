using Distributed; (need = 9 - nworkers()) > 0 && addprocs(need; exeflags="--project=$(dirname(Base.active_project()))"); atexit(() -> try rmprocs(workers()) catch end)
using StatsPlots
@everywhere using BilingualTurk_Julia

mtdata = DataFrame(CSV.File("../Exp2(lab)_forPub/Data/mtdata_long.csv"))[:,:];
subsample = true
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
end
mtdata = subject_to_idx(mtdata)
mtdata = @chain mtdata begin
    @mutate(ang = atan.(ypos, xpos), vot_norm = (VOT + 20)/(60) * 2 - 1)
    @mutate(G = case_when(language == "Monolingual English" => 1,
                          language == "Bilingual English" => 2,
                          language == "Bilingual Spanish" => 3))
    @select(S, trial, G, votstep, ang, mt_seq)
    @pivot_wider(names_from = mt_seq, values_from = ang)
end
S = mtdata.S
G = mtdata.G
V = mtdata.votstep
y = Matrix{Float64}(mtdata[:, 5:end])

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
notConv = chndf[chndf.rhat .> 1.01,:]

#### 

i = 3000
sub = S[i]
grp = G[i]
β₀ = mean(chn_ssmod, "β₀[$grp]")
u_vot = mean(chn_ssmod, "u_vot")
ϵ = [mean(chn_ssmod, "ϵ[$sub, $n]") for n in 1:100]
z = Array{Float64}(undef, 101)
z[1] = mean(chn_ssmod, "z0[$sub]")
for n in 2:101
    z[n] = z[n-1] + ϵ[n-1]
end
plot(1:101, z)

p = Array{Float64}(undef, 9, 101)
i=0
for vot in range(-1, 1, length=9)
    i += 1
    for n in 1:101
        p[i, n] = logistic(β₀ + u_vot * vot + z[n])
    end
end
plot(1:101, p[1,:], color=:red)
plot!(1:101, p[9,:], color=:blue)