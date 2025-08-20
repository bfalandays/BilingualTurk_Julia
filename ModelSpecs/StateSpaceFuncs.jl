module StateSpaceFuncs 
# ================== START MODULE ==================
    export ssmod

    using Reexport
    @reexport using DataFrames, TidierData, CSV, Random, StatsFuns, Distributed, Distributions, Turing, ParetoSmooth, ReverseDiff, Plots, LaTeXStrings, LinearAlgebra, JLD2
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

        β₀ ~ MvNormal(zeros(3), I) # Intercept for each language group

        u_vot ~ Normal(0, 1) # Slope for effect of VOT

        z = Array{Float64}(undef, n_subjects, n_max) # n_subjects x n_timesteps array to store latent variable z
        for i in 1:n_subjects
            z[i, 1] ~ Normal(0, 1) # Prior for latent variable z at time 0
            for n in 2:n_max
                z[i, n] ~ Normal(z[i, n-1], 1) # AR(1) process for latent variable z
            end
        end

        # ϵ ~ filldist(Normal(), n_subjects, n_max-1) # n_subjects x n_timesteps-1 array of innovations for AR(1) process
        # z = Array{Float64}(undef, n_subjects, n_max) # n_subjects x n_timesteps array to store latent variable z
        # z0 ~ filldist(Normal(), n_subjects)
        # z[:, 1] = z0
        # z[:, 2:n_max] = z0 .* ones(1, n_max-1) .+ cumsum(ϵ; dims=2)

        # region ---- LIKELIHOOD ---- #
            for j in 1:size(y, 1) # Outer loop over trials

                for n in 2:n_max # Inner loop over timesteps

                    πₙ = logistic(β₀[G[j]] + u_vot * V[j] + z[S[j], n]) # Compute mixing weight for right-side response option at time n

                    dₙ = y[j, n] < pi/2 ? abs(y[j, n] - μᵣ) : abs(y[j, n] - μₗ) # Compute deviation from idealized trajectory at time n

                    κₙ = scale_exp(dₙ) # Compute concentration parameter κ at time n, based on deviation

                    y[j, n] ~ MixtureModel([VonMises(μᵣ, κₙ), VonMises(μₗ, κₙ)], [πₙ, 1 - πₙ]) # Define likelihood for observed data at time n
                end
            end
        # end
    end


# ================== END MODULE ==================
end