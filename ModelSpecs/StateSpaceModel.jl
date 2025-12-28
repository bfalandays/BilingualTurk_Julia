module StateSpaceModel

#= TO DO / TO THINK ABOUT:
    - Now that the state space model is working, how do I link it to the category model?
        Can I use the output of the category model as the coefficient for VOT?

    - Think through the choice of priors more carefully
    - Consider adding non-linear terms for VOT (e.g. quadratic polynomials or splines)
    - Allow for variability of category boundary across subjects and groups
    - Think through the (potential) hierarchical structure of the model more carefully. 
        - Specifically, should groups be treated as independent?
        - How does the model structure influence the possible inferences?
            - e.g. if I want something like Helmert coding, where I compare bilinguals vs monolinguals, and then within bilinguals compare Eng vs Span,
              how would I do that?
    - Consider a more complex model that includes both x and y positions, instead of collapsing to yrad. How would I model the measurement equation?
        - I could try Tim Meyer's (Meyer, Kim, Spivey & Yoshimi, 2024) approach of using complex-valued time series were zₙ = xₙ + i*yₙ.
            This would allow for 1D data while preserving the full 2D info. But not clear what the measurement equation would look like.
        - Another thing that is wonky to me about the data simplification is that it only considers the angle of the cursor relative to the start position,
            which doesn't incorporate movement direction. So e.g. if the cursor passes the top-right response box and doubles back, the fact that the cursor
            is moving back downwards isnt captured.
        - Similar to above, it doesn't account for distance from the start position. E.g. atan(.1,.1) = atan(1,1). The magnitude info is lost, in addition to direction.
    - Would it make sense to use Δy as the measurement, instead of y? (i.e. change in angle/position, instead of current angle/position)
        - If so, I think that would amount to the states controlling accelerations? So second-order control?
    - Create unit tests for verifying that the Kalman version is indeed approximating the full version
    - Document code more thoroughly
=#

# ================== START MODULE ==================

export prepare_data_SS, 
    expꜛ, 
    𝒢, 
    μᵣ, 
    μₗ, 
    ssmod, 
    plot_zMAP_subj 

using Reexport
@reexport using ..Common 
using SplitApplyCombine, LaTeXStrings, LinearAlgebra, SparseArrays
gr()

# ---- HELPER FUNCS ---- #
function prepare_data_SS(df; subsample = false)
    if subsample 
        println("Subsampling 10 subjects from each group for quicker testing")
        Random.seed!(1)
        sampled_subjects = @chain df begin
            @select(subject, lang_grp)
            @distinct()
            @group_by(lang_grp)
            @slice_sample(n = 10, replace=false)
            @ungroup()
            @arrange(subject)
            @pull(subject)
        end
        df = @chain df @filter(subject in !!sampled_subjects)
    end;
    df = subject_to_idx(df);
    df = @chain df begin
        @mutate(
            yrad = atan.(ypos, xpos), 
            G = case_when(lang_grp == "BE" => 1,
                        lang_grp == "BS" => 2,
                        lang_grp == "ME" => 3))
        @group_by(G)
        @mutate(
            vot_c  = VOT .- mean(VOT),                                   # centered
            vot_z  = (VOT .- mean(VOT)) ./ (std(VOT) + eps())            # z-scored
        )
        @ungroup()                        
        #@filter(votstep == 1 || votstep == 9)
        @select(subject, S, trial, lang_grp, G, VOT, votstep, vot_z, yrad, mt_seq)
        @pivot_wider(names_from = mt_seq, values_from = yrad)
    end;
    S = df.S;
    G = df.G; 
    V = df.VOT; #df.vot_z;
    Y =  Matrix{Float64}(@chain df @select(Symbol("1"):Symbol("101")));
    D = @. ifelse(Y < π/2, abs(Y - μₗ), abs(Y - μᵣ)); # pre-compute the deviation from the far-side response option for all y
    # D = @. ifelse(Y > π/2, abs(Y - μₗ), abs(Y - μᵣ)); # pre-compute the deviation from the far-side response option for all y
    Κ = expꜛ(D); # pre-compute the concentration parameter for all Y 

    J = length(S); I = maximum(S); cnts = counts(S, 1:I)                # trials per subject
    rows = S; cols = 1:J; vals = 1.0 ./ cnts[rows]               # weight = 1 / n_trials_subject
    Am = sparse(rows, cols, vals, I, J)   # ::SparseMatrixCSC{Float64,Int}

    return S, G, V, Y, D, Κ, Am, df
end

# ---------------------- #
function expꜛ(d::AbstractMatrix; lb::Real=10.0, ub::Real=200.0)
    d¹ = exp.(d)
    mins = minimum(d¹, dims=1); maxs = maximum(d¹, dims=1)
    d² = lb .+ ((d¹ .- mins) ./ (maxs .- mins)) .* (ub - lb)
    
    return d² #
    # return sqrt.( 1 ./ d²)
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

    Link function from latent states (z)/parameters (β) to measurements Y.  

    `form` can be:
    - `:logistic` (default):  π / (1 + exp(-β * z))
    - `:gompertz``:           π * exp(-β * exp(-z))

    Normally, this function is π-scaled, such that the output is in the range (0, π].
    But when it is normalized to the range [0,1], it can be interpreted as the probability of navigating closer to the distractor option.

    So the `scale` argument can be set to:
    - `:π` (default): output in range (0, π]
    - `:unit`: output in range [0, 1]
"""
function 𝒢(
    z::Union{Real,AbstractArray{<:Real}}, 
    β::Union{Real,AbstractArray{<:Real}}; 
    lb::Real=.1, 
    form::Symbol=:logistic, 
    scale::Symbol=:π)
    
    if scale == :π
        ub = π - lb; 
    elseif scale == :unit
        ub = 1.0 - lb; 
    else
        throw(ArgumentError("`scale` must be :π or :unit"))
    end
    R = ub - lb;
    
    if form == :logistic
        # return @. lb + (R / (1 + exp(β - z)))   
        return @. lb + (R * logistic(z - β))      
    elseif form == :gompertz
        return @. lb + (R * exp(-β * exp(-z)))
    else
        throw(ArgumentError("`form` must be :logistic or :gompertz"))
    end
end

const μᵣ = atan(1114, 800); # angle of right-side response option
const μₗ = atan(1114, -800); # angle of left-side response option
const LOG2π = log(2π)

# STATE SPACE MODEL 
@model function ssmod(
    S::AbstractVector{<:Integer}, # Length J, indexes the subject for trial j
    G::AbstractVector{<:Integer}, # Length J, indexes the language group of subject on trial j 
    V::AbstractVector{<:Real}, # Length J, the stimulus VOT value (scaled to -1:1) on trial j
    Y::AbstractMatrix{<:Real}, # JxN, the observed angle on trial j at timestep n
    R::AbstractMatrix{<:Real}, # J×N, measurement variance term proxy (1/√Kappa)
    Am::SparseMatrixCSC{Float64, Int64}, # I×J subject-averaging matrix
    ::Type{T}=Float64) where {T} # ensuring types can be inferred

    n_groups = maximum(G); 
    contrasts = [0 0; 1 0; 0 1]
    n_subjects = maximum(S)
    N = size(Y, 2);
    J = size(Y, 1);

    #= Note: do I need to use absolute distance? 
        Feels like for the DDM I could use signed distance, because the sign indicates direction of evidence accumulation.
        But here, since β is a coefficient for the logistic function that determines attraction towards either response option,
        it makes more sense to use absolute distance from the category boundary?
    =#

    # Boundary for each group
    θ ~ MvNormal(fill(10.0, n_groups), 5.0^2 * I)
    ΔV = abs.(V - θ[G])

    # Global slope for ΔV
    η ~ Normal(0.0, 1.0); # η = rand(Normal())

    # VOT x Group interaction term
    δ ~ MvNormal(zeros(n_groups-1), 1.0^2 * I) # δ = rand(MvNormal(zeros(n_groups-1), I))
    δⱼ = (contrasts[G,:] * δ)
    
    βⱼ = ΔV .* (η .+ δⱼ)

    ẑ = Matrix{T}(undef, n_subjects, N); ẑ[:, 1] .= zero(T)
    λ̂ = Matrix{T}(undef, n_subjects, N); λ̂[:, 1] .= one(T)
    Q = one(T);
    
    tmpJ1 = Vector{T}(undef, J)  # G or residual*G
    tmpJ2 = Vector{T}(undef, J)  # G.^2 .* σ
    tmpI  = Vector{T}(undef, n_subjects) 
    @inbounds for n in 1:N # Loop over timesteps
        ycol = @view Y[:, n]
        rcol = @view R[:, n]

        z̅ = (n == 1) ? ẑ[:, 1] : ẑ[:, n-1]
        λ̅ = (n == 1) ? λ̂[:, 1] : (λ̂[:, n-1] .+ Q)
        
        ŷ = 𝒢(z̅[S], βⱼ; form=:logistic)
        σ = λ̅[S] + R[:, n]

        # -- LIKELIHOOD -- #
        # Y[:,n] ~ MvNormal(ŷ, Diagonal(σ))
        
        # Y[:,n] ~ arraydist(Normal.(ŷ, sqrt.(σ)))

        s = zero(eltype(ycol)) # s = zero(T)
        @inbounds @simd for j in eachindex(ycol, ŷ, σ)
            r = ycol[j] - ŷ[j]
            s += LOG2π + log(σ[j]) + (r*r)/σ[j]
        end
        Turing.@addlogprob!(-0.5*s)

        # -- Kalman update step --#
        # K = λ̅[S] ./ σ # Kalman gain term
        # ẑ[:, n] = z̅ + Am * ((Y[:, n] .-  ŷ) .* K) 
        # λ̂[:, n] = λ̅ - Am * (K .* σ .* K)
    
        @. tmpJ1 = λ̅[S] / σ                  # K
        @. tmpJ2 = (tmpJ1 * tmpJ1) * σ       # K.^2 .* σ
        @. tmpJ1 = (ycol - ŷ) * tmpJ1       # (y - ŷ) .* K
        mul!(tmpI, Am, tmpJ1)                  # Am * ((y - ŷ) .* K)
        @. ẑ[:, n] = z̅ + tmpI
        mul!(tmpI, Am, tmpJ2)                  # Am * (K.^2 .* σ)
        @. λ̂[:, n] = λ̅ - tmpI
    end

    zsubj := ẑ

    # # Backwards smoothing pass
    # s_ẑ = copy(ẑ); s_λ̂ = copy(λ̂);
    # @inbounds for n in (N-1):-1:1
    #     # forward-side predictions for time nn+1
    #     s_z̅ = s_ẑ[:, n]
    #     s_λ̅ = s_λ̂[:, n] .+ Q

    #     # smoothing gain
    #     s_K = s_λ̂[:,n] ./ s_λ̅

    #     # smoothed updates
    #     s_ẑ[:,n] = s_ẑ[:,n] .+ s_K .* (s_ẑ[:,n+1] .- s_z̅)
    #     s_λ̂[:,n] = s_λ̂[:,n] .+ s_K .* (s_λ̂[:,n+1] .- s_λ̅) .* s_K
    # end

    # zsubj := s_ẑ
end

# ------------------------------------------------------------

function plot_zMAP_subj(chn, S, G, V, s::Integer; N::Int=100, form=:logistic)
    
    zˢ = [mean(chn, "zsubj[$s, $n]") for n in 1:N]
    g = G[S .== s][1];
    V = sort(unique(V[S .== s]))
    contrasts = [0 0; 1 0; 0 1]
    η = mean(chn[:η])
    δ = contrasts[g, :]' * [mean(chn, "δ[$g]") for g in 1:2]
    #β = V .* (η + δ)

    

    θ = mean(chn, "θ[$g]")
    Vc = abs.(V .- θ) # center VOT based on group boundary
    β = Vc .* (η + δ) # β is a vector of coefficients for the logistic function, varying by group and VOT

    zᵛ = Array{Float64}(undef, length(V), N);
    for (i,v) in enumerate(V)
        zᵛ[i,:] = [𝒢(zˢ[n], β[i]; form=form, scale=:unit) for n in 1:N]
    end

    μ = Array{Float64}(undef, length(V), length(-5.0:.1:5.0));
    for (i,v) in enumerate(V)
        μ[i,:] = [𝒢(z, β[i]; form=form, scale=:unit) for z in -5.0:.1:5.0]
    end

    layout = @layout (3,1);
    layout = @layout([[a ; b] c{.4w}])
    fig = plot(layout=layout, size=(720,420), left_margin=5Plots.mm, plot_title="Subj = $s, Grp = $g");
    plot!(fig[1], 1:N, zˢ, 
        label=nothing,
        xlabel="timestep",
        ylabel="Z",
        color=:black
    );
    hline!(fig[1],[0], linestyle=:dash, label=nothing, color=:black);

    for (i,v) in enumerate(V)#1:9
        plot!(fig[2], 1:N, zᵛ[i, :], 
            label=["Step $i"],
            xlabel="timestep",
            ylabel="/p/ <---> /b/"
            #ylims=(-.5,.5)
        );
    end
    hline!(fig[2],[.5], linestyle=:dash, label=nothing);

    #colors9 = cgrad(:viridis, 9, categorical=true)
    for (i,v) in enumerate(V)
        plot!(fig[3], -5.0:.1:5.0, μ[i,:], 
            label=["Step $i"],
            xlabel="Z",
            ylabel="μ"#,
            #c = colors9[i]
        );
    end
    hline!(fig[3],[.5], linestyle=:dash, label=nothing);

    # display(fig)

    return fig
end
# ================== END MODULE ==================
end