module StateSpaceFuncs 
# ================== START MODULE ==================
export ssmod, plotSS, getMAP, plotSubjMAP, ssmod_kalman

using Reexport
@reexport using SplitApplyCombine, DataFrames, TidierData, CSV, Random, StatsFuns, StatsBase, Distributed, Distributions, Turing, ParetoSmooth, ReverseDiff, Plots, LaTeXStrings, LinearAlgebra, JLD2
gr()

# ---- HELPER FUNCS ---- #
function dviz(d) # convenience function for visualizing distributions
    tmp = rand(d,10000)
    if typeof(d) <: Beta
        display(histogram(tmp, xlims=(0,1)))
    else
        display(histogram(tmp))
    end
    
    return mean(tmp), std(tmp)
end

function subject_to_idx(data)
    d = Dict(s => i for (i, s) in enumerate(unique(data.subject)))
    data.S = [d[s] for s in data.subject]
    return data
end 

stimContinuum = collect(range(-20, 40, length=9));

# ---------------------- #
function expꜛ(d::AbstractMatrix; lb::Real=10.0, ub::Real=200.0)
    d¹ = exp.(d)
    mins = minimum(d¹, dims=1); maxs = maximum(d¹, dims=1)
    d² = lb .+ ((d¹ .- mins) ./ (maxs .- mins)) .* (ub - lb)
    return sqrt.( 1 ./ d²)
end

#= π-scaled function. 
    Following the code in the ssMousetrack R package, they set a lower bound of .1, and upper bound of π-.1 "to avoid the case y[n]~0". 
    See ssMousetrack/R/run_ssm.R, line 76 and 86, where `bnds` is initialized.
    Also see ssMousetrack/inst/stan/fit_model_log.stan, line 83 (logistic version)
    or ssMousetrack/inst/stan/fit_model_gomp.stan, line 87 (gompertz version),
    which is where the bounds are used.
=#
"""
    𝒢(z, β; lb = .1, opt = :logistic)

π-scaled link function.  
`opt` can be:
- `:logistic` (default):  π / (1 + exp(-β * z))
- `:gompertz``:           π * exp(-β * exp(-z))
"""
function 𝒢(z::AbstractVector{<:Real}, β::AbstractVector{<:Real}; lb::Real=.1, opt::Symbol=:logistic)
    ub = π - lb; R = ub - lb;
    if opt == :logistic
        return lb .+ R .* logistic.(β .- z)
    elseif opt == :gompertz
        return lb .+ ((ub - lb) .* exp.(-β .* exp.(-z)))
    else
        throw(ArgumentError("opt must be :logistic or :gompertz"))
    end
end

const μᵣ = atan(1046, 778); # angle of right-side response option
const μₗ = atan(1046, -778); # angle of left-side response option

# STATE SPACE MODEL 
@model function ssmod( 
    S::AbstractVector{<:Integer}, # S ϵ ℝᴶ ; S[j] indexes the subject for trial j
    G::AbstractVector{<:Integer}, # G ϵ ℝᴶ ; G[j] indexes the language group to which subject S[j] belongs
    V::AbstractVector{<:Real}, # V ϵ ℝᴶ ; V[j] is the stimulus VOT value (scaled to -1:1) on trial j
    Y::AbstractMatrix{<:Real}) # y ϵ ℝᴶˣᴺ ; y[j, n] is the observed angle on trial j at timestep n
        #= NOTE:
            Because I filtered out trials with extreme response times, I don't have equal numbers of trials for each subject.
            As such, I can't format the data in a 3D array, so instead I have a 2D array where each row is a unique trial and 
            columns are timesteps within a trial.
        =#

    n_subjects = length(unique(S))
    n_groups = length(unique(G))
    Gmat = indicatormat(G)'

    γₖ ~ MvNormal(zeros(n_groups), I)
    η ~ Normal();
    δₖ ~ MvNormal(zeros(n_groups), I)
    βⱼ = Gmat*γₖ + V*η + (V .* (Gmat * δₖ)) # β is a vector of coefficients for the logistic function, varying by group and VOT

    zₛₓₙ = Array{Float64}(undef, n_subjects, size(Y, 2)) # n_subjects x n_timesteps array to store latent variable z
    for s in 1:n_subjects
        zₛₓₙ[s, 1] ~ Normal(0, 1) # Prior for latent variable z at time 0
        for n in 2:size(Y, 2)
            zₛₓₙ[s, n] ~ Normal(zₛₓₙ[s, n-1], 1) # AR(1) process for latent variable z
        end
    end

    Dⱼₓₙ = @. ifelse(Y < π/2, abs(Y - μₗ), abs(Y - μᵣ)) # pre-compute the deviation from the far-side response option for all y
    Κⱼₓₙ = expꜛ(Dⱼₓₙ) # pre-compute the concentration parameter for all y 
    # for j in 1:size(y, 1) # Outer loop over trials
    #     for n in axes(Y,2) # Inner loop over timesteps
    #         μⱼₓₙ = 𝒢(zₛₓₙ[S[j], n], βⱼ) 
    #         Y[j, n] ~ VonMises(μⱼₓₙ, κⱼₓₙ[j,n])
    #     end
    # end

    for n in axes(Y,2) # Inner loop over timesteps
        μⱼ = 𝒢(zₛₓₙ[S, n], βⱼ) 
        Y[:, n] ~ arraydist(VonMises.(μⱼ, Κⱼₓₙ[:,n]))
    end  
end

function getMAP(chndict, mtdata)
    n_subjects = length(unique(mtdata.S))
    n_groups = length(unique(mtdata.G))
    V = sort(unique(mtdata.vot_norm))

    γₖ = [chndict[Symbol("γₖ[$g]")] for g in 1:n_groups]
    η = chndict[:η]
    δₖ = [chndict[Symbol("δₖ[$g]")] for g in 1:n_groups]

    zₛₙ = Array{Float64}(undef, n_subjects, 101)
    for s in 1:n_subjects
        g = mtdata[mtdata.S .== s, :G][1]
        for n in 1:101
            zₛₙ[s, n] = chndict[Symbol("zₛₙ[$s, $n]")]
        end
    end

    f(g, v, z) = logistic((γₖ[g] + V[v]*η + (V[v] * δₖ[g])) + z)
    πlogistic = f

    return zₛₙ, πlogistic
end

function plotSubjMAP(zₛₙ, πlogistic, mtdata, sidx)
    
    subj = mtdata[mtdata.S .== sidx, :subject][1];
    g = mtdata[mtdata.S .== sidx, :G][1];

    πᵥₙ = Array{Float64}(undef, 9, 101)
    zrng = -5:.1:5
    πᵥ = Array{Float64}(undef, 9, length(zrng))
    for v in 1:9
        πᵥₙ[v, :] = πlogistic.(g, v, zₛₙ[sidx, :])
        πᵥ[v, :] = πlogistic.(g, v, zrng)
    end

    layout = @layout (3,1);
    layout = @layout([[a ; b] c{.4w}])
    fig = plot(layout=layout, size=(720,420), left_margin=5Plots.mm, plot_title="Subj = $subj, Grp = $g");
    plot!(fig[1], 1:101, zₛₙ[sidx, :], 
        label=nothing,
        xlabel="timestep",
        ylabel="Z",
        color=:black
    );
    hline!(fig[1],[0], linestyle=:dash, label=nothing, color=:black);

    for i in [1, 9]#1:9
        plot!(fig[2], 1:101, πᵥₙ[i, :], 
            label=["Step $i"],
            xlabel="timestep",
            ylabel="π"
        );
    end
    hline!(fig[2],[.5], linestyle=:dash, label=nothing);

    #colors9 = cgrad(:viridis, 9, categorical=true)
    for i in [1, 9]
        plot!(fig[3], zrng, πᵥ[i,:], 
            label=["Step $i"],
            xlabel="Z",
            ylabel="π"#,
            #c = colors9[i]
        );
    end
    hline!(fig[3],[.5], linestyle=:dash, label=nothing);

    # display(fig)

    return fig
end

function subjmeans(Q::Vector{Float64}, S::Vector{Int})
    @assert length(Q) == length(S)
    K = maximum(S)
    sums   = zeros(Float64, K)
    counts = zeros(Int, K)
    @inbounds @simd for i in eachindex(Q,S)
        g = S[i]
        sums[g]   += Q[i]
        counts[g] += 1
    end
    sums ./ counts
end
# @btime tmp = subjmeans(Q₁, S) # 46.166 μs (6 allocations: 912 bytes)

@model function ssmod_kalman(
    S::AbstractVector{<:Integer},
    G::AbstractVector{<:Integer},
    V::AbstractVector{<:Real},
    Y::AbstractMatrix{<:Real})

    n_subjects = length(unique(S))
    n_groups = length(unique(G))
    Gmat = indicatormat(G)'

    γₖ ~ MvNormal(zeros(n_groups), I) # γₖ = rand(MvNormal(zeros(n_groups), I))
    η ~ Normal(); # η = rand(Normal())
    δₖ ~ MvNormal(zeros(n_groups), I) # δₖ = rand(MvNormal(zeros(n_groups), I))
    βⱼ = Gmat*γₖ + V*η + (V .* (Gmat * δₖ)) # β is a vector of coefficients for the logistic function, varying by group and VOT

    ẑₛₓₙ = Array{Float64}(undef, n_subjects, size(Y, 2)); ẑₛₓₙ[:, 1] .= 1e-04; z̅ₛₓₙ = similar(ẑₛₓₙ); z̅ₛₓₙ[:, 1] .= 1e-04;
    λ̂ₛₓₙ = Array{Float64}(undef, n_subjects, size(Y, 2)); λ̂ₛₓₙ[:, 1] .= 1.0; λ̅ₛₓₙ = similar(λ̂ₛₓₙ); λ̅ₛₓₙ[:, 1] .= 1.0;
    ŷⱼₓₙ = Array{Float64}(undef, size(Y, 1), size(Y, 2));
    σⱼₓₙ = Array{Float64}(undef, size(Y, 1), size(Y, 2));

    Dⱼₓₙ = @. ifelse(Y < π/2, abs(Y - μₗ), abs(Y - μᵣ)) # pre-compute the deviation from the far-side response option for all y
    Κⱼₓₙ = expꜛ(Dⱼₓₙ) # pre-compute the concentration parameter for all y 
    for n in axes(Y,2)#1:size(y, 2) # Inner loop over timesteps κ
        if n > 1
            z̅ₛₓₙ[:, n] = ẑₛₓₙ[:, n-1]
            λ̅ₛₓₙ[:, n] = λ̂ₛₓₙ[:, n-1] .+ 1.0
        end
        z̅ⱼ = z̅ₛₓₙ[S, n]
        λ̅ⱼ = λ̅ₛₓₙ[S, n]

        ŷⱼₓₙ[:,n] = 𝒢(z̅ⱼ, βⱼ)

        σⱼₓₙ[:, n] = λ̅ⱼ + Κⱼₓₙ[:, n]

        #= Benchmarking different ways to do the inference step
            using BenchmarkTools
            μ   = copy(ŷⱼₙ[:, n])
            σ²  = copy(σⱼₙ[:, n])
            σ   = sqrt.(σ²)
            yv  = copy(y[:, n])
            @btime sum(logpdf.(Normal.($μ, $σ), $yv)); # 62.167 μs (6 allocations: 128.16 KiB)
            @btime logpdf(arraydist(Normal.($μ, $σ)), $yv); # 68.166 μs (3 allocations: 224.06 KiB)
            @btime logpdf(MvNormal($μ, Diagonal($σ²)), $yv); # 66.375 μs (3 allocations: 128.06 KiB)
        =#
        Y[:,n] ~ MvNormal(ŷⱼₓₙ[:,n], Diagonal(σⱼₓₙ[:, n]))

        Gₙ = λ̅ⱼ ./ σⱼₓₙ[:, n] # Kalman gain term
        ẑₛₓₙ[:, n] = z̅ₛₓₙ[:, n] + subjmeans(((Y[:,n] .-  ŷⱼₓₙ[:,n]) .* Gₙ), S)
        λ̂ₛₓₙ[:, n] = λ̅ₛₓₙ[:, n] - subjmeans((Gₙ .* σⱼₓₙ[:, n] .* Gₙ), S)
        #=
            @btime tmp = [mean(@view Q[S .== s]) for s in unique(S)] # 143.708 μs (285 allocations: 166.33 KiB)
            @btime tmp2 = combine(groupby(DataFrame(S = S, Q = Q), :S), :Q => mean => :Δz)[!,:Δz] # 124.291 μs (320 allocations: 400.44 KiB)
            @btime tmp4 = map(mean, SplitApplyCombine.group(S, Q)).values # 81.375 μs (212 allocations: 242.89 KiB)
            @btime tmp5 = map(s -> mean(Q[S .== s]), unique(S)) #  136.459 μs (264 allocations: 165.27 KiB)
        =#
    end
end



# ================== END MODULE ==================
end