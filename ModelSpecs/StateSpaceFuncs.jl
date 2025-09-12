module StateSpaceFuncs 
# ================== START MODULE ==================
export ssmod, plotSS, getMAP, plotSubjMAP

using Reexport
@reexport using DataFrames, TidierData, CSV, Random, StatsFuns, StatsBase, Distributed, Distributions, Turing, ParetoSmooth, ReverseDiff, Plots, LaTeXStrings, LinearAlgebra, JLD2
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

# "Exponential function scaled in the natural range of the concentration parameter (e.g. lb = 10, ub = 200)"
function scale_exp(d; lb=10.0, ub=200.0)
    raw = @. exp(d)
    rmin, rmax = 1.0, exp(pi/4) 
    κ = @. lb + (raw - rmin) * (ub - lb) / (rmax - rmin)
    return κ
end

# STATE SPACE MODEL 
@model function ssmod( 
    S::AbstractVector{<:Real}, # S ϵ ℝᴶ ; S[j] indexes the subject for trial j
    G::AbstractVector{<:Real}, # G ϵ ℝᴶ ; G[j] indexes the language group to which subject S[j] belongs
    V::AbstractVector{<:Real}, # V ϵ ℝᴶ ; V[j] is the stimulus VOT value (scaled to -1:1) on trial j
    y::AbstractMatrix{<:Real}) # y ϵ ℝᴶˣᴺ ; y[j, n] is the observed angle on trial j at timestep n
        #= NOTE:
            Because I filtered out trials with extreme response times, I don't have equal numbers of trials for each subject.
            As such, I can't format the data in a 3D array, so instead I have a 2D array where each row is a unique trial and 
            columns are timesteps within a trial.
        =#

    n_subjects = length(unique(S))
    n_groups = length(unique(G))
    n_max = 101 # Each trial is time-normalized to 101 timesteps
    Gmat = indicatormat(G)'

    μᵣ = atan(1046, 778); # angle of right-side response option
    μₗ = atan(1046, -778); # angle of left-side response option
    # κ = 200.0; # concentration parameter (not sure if it should be fixed here or set dynamically as in the likelihood loop below?)

    γₖ ~ MvNormal(zeros(n_groups), I)
    η ~ Normal();
    δₖ ~ MvNormal(zeros(n_groups), I)
    βⱼ = Gmat*γₖ + V*η + (V .* (Gmat * δₖ)) # β is a vector of coefficients for the logistic function, varying by group and VOT

    zₛₙ = Array{Float64}(undef, n_subjects, n_max) # n_subjects x n_timesteps array to store latent variable z
    for i in 1:n_subjects
        zₛₙ[i, 1] ~ Normal(0, 1) # Prior for latent variable z at time 0
        for n in 2:n_max
            zₛₙ[i, n] ~ Normal(zₛₙ[i, n-1], 1) # AR(1) process for latent variable z
        end
    end

    # V1 
    for j in 1:size(y, 1) # Outer loop over trials
        for n in 2:n_max # Inner loop over timesteps
            πⱼₙ = logistic(βⱼ[j] + zₛₙ[S[j],n]) # Compute mixing weight for right-side response option at time n
            dⱼₙ = y[j, n] < pi/2 ? abs(y[j, n] - μᵣ) : abs(y[j, n] - μₗ) # Compute deviation from idealized trajectory at time n
            κⱼₙ = scale_exp(dⱼₙ) # Compute concentration parameter κ at time n, based on deviation
            y[j, n] ~ MixtureModel([VonMises(μᵣ, κⱼₙ), VonMises(μₗ, κⱼₙ)], [πⱼₙ, 1 - πⱼₙ]) # Define likelihood for observed data at time n
        end
    end
    
    # # ## V2 
    # πⱼₙ = logistic.(βⱼ .+ zₛₙ[S, :]) # Compute mixing weight for right-side response option at each time step
    # dⱼₙ = ifelse.(Matrix{Bool}(y .< pi/2), abs.(y .- μᵣ), abs.(y .- μₗ))
    # κⱼₙ = scale_exp.(dⱼₙ)
    # dists = map((a,b) -> [a,b], VonMises.(μᵣ, κⱼₙ), VonMises.(μₗ, κⱼₙ))
    # weights = map((a,b) -> [a,b], πⱼₙ, 1 .- πⱼₙ)
    # y ~ product_distribution(MixtureModel.(dists, weights)) # Define likelihood for observed data
    # # for j in 1:size(y, 1)
    # #     for n in 1:n_max
    # #         y[j, n] ~ MixtureModel([VonMises(μᵣ, κⱼₙ[j, n]), VonMises(μₗ, κⱼₙ[j, n])], [πⱼₙ[j, n], 1 - πⱼₙ[j, n]])
    # #     end
    # # end
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

# ================== END MODULE ==================
end