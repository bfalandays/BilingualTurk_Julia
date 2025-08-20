module StateSpaceFuncs 
# ================== START MODULE ==================
export ssmod, plotSS

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
function scale_exp(dₙ; lb=10.0, ub=200.0)
    raw = @. exp(dₙ)
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
    n_max = 101 # Each trial is time-normalized to 101 timesteps

    μᵣ = atan(1046, 778); # angle of right-side response option
    μₗ = atan(1046, -778); # angle of left-side response option

    # κ = 200.0; # concentration parameter (not sure if it should be fixed here or set dynamically as in the likelihood loop below?)

    β₀ ~ MvNormal(zeros(3), I) # Intercept term for each level of language group (G)

    β₁ ~ MvNormal(zeros(3), I) # Normal() # Slope for effect of VOT (V), varying by group to encode interaction

    z = Array{Float64}(undef, n_subjects, n_max) # n_subjects x n_timesteps array to store latent variable z
    for i in 1:n_subjects
        z[i, 1] ~ Normal(0, 1) # Prior for latent variable z at time 0
        for n in 2:n_max
            z[i, n] ~ Normal(z[i, n-1], 1) # AR(1) process for latent variable z
        end
    end

    # region ---- LIKELIHOOD ---- #
        for j in 1:size(y, 1) # Outer loop over trials

            for n in 2:n_max # Inner loop over timesteps

                πₙ = logistic(β₀[G[j]] + β₁[G[j]] * V[j] + z[S[j],n]) # Compute mixing weight for right-side response option at time n

                dₙ = y[j, n] < pi/2 ? abs(y[j, n] - μᵣ) : abs(y[j, n] - μₗ) # Compute deviation from idealized trajectory at time n

                κₙ = scale_exp(dₙ) # Compute concentration parameter κ at time n, based on deviation

                y[j, n] ~ MixtureModel([VonMises(μᵣ, κₙ), VonMises(μₗ, κₙ)], [πₙ, 1 - πₙ]) # Define likelihood for observed data at time n
            end
        end
    # end
end


function plotSS(chain, data, S)
    
    G = unique(data.G[data.S .== S])[1]
    β₀ = mean(chain, "β₀[$G]")
    u_vot = mean(chain, "u_vot")

    z = Array{Float64}(undef, 101)
    for n in 1:101
        z[n] = mean(chain, "z[$S, $n]")
    end

    πᵥₙ = Array{Float64}(undef, 9, 101)
    for (i,V) in enumerate(sort(unique(data.votstep)))
        for n in 1:101
            πᵥₙ[i, n] = logistic(β₀ + u_vot * V + z[n])
        end
    end

    π_z = Array{Float64}(undef, 9, 100)
    for (i,V) in enumerate(sort(unique(data.votstep)))
        for (j,z_) in enumerate(range(-5, 5, length=100))
            π_z[i, j] = logistic(β₀ + u_vot * V + z_)
        end
    end

    layout = @layout (3,1);
    layout = @layout([[a ; b] c{.4w}])
    fig = plot(layout=layout, size=(720,420), left_margin=5Plots.mm);
    plot!(fig[1], 1:101, z, 
        label=nothing,
        xlabel="timestep",
        ylabel="Z",
        color=:black
    );
    hline!(fig[1],[0], linestyle=:dash, label=nothing);

    plot!(fig[2], 1:101, πᵥₙ[[1,9],:]', 
        label=["step1" "step9"],
        xlabel="timestep",
        ylabel="π"
    );
    hline!(fig[2],[.5], linestyle=:dash, label=nothing);

    colors9 = cgrad(:viridis, 9, categorical=true)
    for i in 1:9
        plot!(fig[3], range(-5,5, 100), π_z[i,:], 
            label=nothing, #["step1" "step9"],
            xlabel="Z",
            ylabel="π",
            c = colors9[i]
        );
    end
    hline!(fig[3],[.5], linestyle=:dash, label=nothing);

    display(fig)
end

# ================== END MODULE ==================
end