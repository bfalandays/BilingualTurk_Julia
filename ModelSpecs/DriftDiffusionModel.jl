module DriftDiffusionModel
# ================== START MODULE ==================

export prepare_data_DDM, 
    DDM,
    DDMhier,
    catDDM,
    posterior_predictive_plot

using Reexport
@reexport using ..Common

# region ================== adapted from SequentialSamplingModels.jl ==================

    struct DDM{T <: Real} <: Distribution{Univariate,Continuous} 
        ν::T
        α::T
        z::T
        τ::T
        function DDM{T}(ν::T, α::T, z::T, τ::T; check_args::Bool = true) where {T <: Real}
            Distributions.@check_args(
                DDM, 
                (α, α > 0.0),
                (z, 0.0 < z < 1.0),
                (τ, τ >= 0)
            )
            return new{T}(ν, α, z, τ)
        end
    end

    Distributions.minimum(d::DDM) = d.τ
    Distributions.maximum(d::DDM) = Inf

    # Constructor
    function DDM(ν::Real, α::Real, z::Real, τ::Real; check_args::Bool=true)
        ν1, α1, z1, τ1 = promote(float(ν), float(α), float(z), float(τ))
        return DDM{typeof(ν1)}(ν1, α1, z1, τ1; check_args=check_args)
    end

    # Keyword constructor 
    function DDM(; ν::Real, α::Real, z::Real, τ::Real, check_args::Bool=true)
        return DDM(ν, α, z, τ; check_args=check_args)
    end

    StatsBase.params(d::DDM) = (d.ν, d.α, d.z, d.τ)

    @inline function Distributions.pdf(d::DDM{T}, choice::Integer, rt::Real; ϵ::Real = 1.0e-4) where {T <: Real}
        rtT = oftype(d.ν, rt); ϵT = oftype(d.ν, ϵ)
        # d.τ < rtT || return zero(T) # these checks shouldn't be doing anything, since we already ensure τ > 0 and rt - τ > 0
        # d.τ >= 0 || return zero(T)
        (ν, α, z, τ) = params(d)

        #= Note:
            When choice == 0 (lower boundary, assigned to /b/), we use the lower boundary pdf.
            When choice == 1 (upper boundary, assigned to /p/), we reflect the parameters
        =#
        # return choice == 0 ? _pdf(ν, α, z, τ, rtT; ϵ=ϵT) : _pdf(-ν, α, one(T) - z, τ, rtT; ϵ=ϵT) 

        c = oftype(d.ν, choice)
        ν_eff = (one(T) - T(2) * c) * ν
        z_eff = (one(T) - c) * z + c * (one(T) - z)
        return _pdf(ν_eff, α, z_eff, τ, rtT; ϵ = ϵT)
    end

    # Probability density function over the lower boundary
    @inline function _pdf(ν::T, α::T, z::T, τ::T, rt::T; ϵ::T) where {T <: Real}

        u = (rt - τ) / α^2 #use normalized time
        piT = oftype(u, π)

        K_s = T(2)
        K_l = one(T) / (piT * sqrt(u))
        
        # number of terms needed for large-time expansion
        if (piT * u * ϵ) < one(T)
            K_l = max(
                sqrt((-T(2) * log(piT * u * ϵ)) / (piT^2 * u)), 
                K_l
            )
        end
        # number of terms needed for small-time expansion
        if (T(2) * sqrt(T(2) * piT * u) * ϵ) < one(T)
            K_s = max( 
                T(2) + sqrt( -2u * log( T(2) * ϵ * sqrt(T(2) * piT * u))), 
                sqrt(u) + one(T) 
            )
        end

        p = exp((-α * z * ν) - (T(0.5) * (ν^2) * (rt - τ))) / (α^2)

        # decision rule for infinite sum algorithm
        return K_s < K_l ? p * _small_time_pdf(u, z, ceil(Int, K_s)) : p * _large_time_pdf(u, z, ceil(Int, K_l))
    end

    # small-time expansion
    @inline function _small_time_pdf(u::T, z::T, K::Int) where {T <: Real}
        inf_sum = zero(T)

        k_series = (-floor(Int, 0.5 * (K - 1))):ceil(Int, 0.5 * (K - 1))
        for k ∈ k_series
            inf_sum += ((2k + z) * exp(-((2k + z)^2 / (2u))))
        end

        # @inbounds for k_int in -K:K
        #     k = T(k_int)
        #     inf_sum += ((2k + z) * exp(-((2k + z)^2 / (2u))))
        # end
        return inf_sum / sqrt(2T(π) * u^3)
    end

    # large-time expansion
    @inline function _large_time_pdf(u::T, z::T, K::Int) where {T <: Real}
        inf_sum = zero(T)

        for k ∈ 1:K
            inf_sum += (k * exp(-0.5 * (k^2 * π^2 * u)) * sin(k * π * z))
        end
        # inf_sum = max(inf_sum, zero(T))

        # @inbounds for k_int in 1:K
        #     k = T(k_int)
        #     inf_sum += (k * exp(-0.5 * (k^2 * π^2 * u)) * sin(k * π * z))
        # end

        return T(π) * inf_sum
    end

    function Distributions.logpdf(d::DDM{T}, choice::Integer, rt::Real; ϵ::Real = 1.0e-4) where {T <: Real}
        log(pdf(d, choice, rt; ϵ = ϵ))
        
        # val = pdf(d, choice, rt; ϵ = ϵ)
        # δ   = eps(T)
        # val_pos = max(val, δ)    # floor at tiny positive to avoid log(0) / log(negative)
        # return log(val_pos)
    end

    Distributions.rand(rng::Distributions.AbstractRNG, d::DDM) = _rand_rejection(rng, d)

    ## n draws (return a NamedTuple of vectors: (choice = Vector{Int}, rt = Vector{Float64}))
    function Distributions.rand(rng::Distributions.AbstractRNG, d::DDM, n::Int)
        choices = Vector{Int}(undef, n)
        rts     = Vector{Float64}(undef, n)
        @inbounds for i in 1:n
            sample     = Distributions.rand(rng, d)
            choices[i] = sample.choice
            rts[i]     = sample.rt
        end
        return (choice = choices, rt = rts)
    end
  
    # Rejection-based Method for the Symmetric Wiener Process(Tuerlinckx et al., 2001 based on Lichters et al., 1995)
    # adapted from the RWiener R package, note, here σ = 0.1
    @inline function _rand_rejection(rng::Distributions.AbstractRNG, d::DDM)
        (ν, α, z, τ) = params(d)

        ϵ = 1.0e-15

        D = 0.005 # = 2*σ^2 => 1/200
        zn = (z * α) / 10 # absolute bias!
        αn = α / 10
        νn = ν / 10

        total_time = 0.0
        start_pos = 0.0
        Aupper = αn - zn
        Alower = -zn
        radius = min(abs(Aupper), abs(Alower))
        λ = 0.0
        F = 0.0
        prob = 0.0

        while true
            if νn == 0
                λ = (0.25D * π^2) / (radius^2)
                F = 1.0
                prob = 0.5
            else
                λ = ((0.25 * νn^2) / D) + ((0.25 * D * π^2) / (radius^2))
                F = (D * π) / (radius * νn)
                F = F^2 / (1 + F^2)
                prob = exp((radius * νn) / D)
                prob = prob / (1 + prob)
            end

            r = rand(rng)
            dir = r < prob ? 1 : -1
            l = -1.0
            s1 = 0.0
            s2 = 0.0

            # Tuerlinckx et al. (2001; eq. 16)  
            while s2 > l
                s1 = rand(rng)
                s2 = rand(rng)
                tnew = 0.0
                tδ = 0.0
                uu = zero(Int)

                while (abs(tδ) > ϵ) || (uu == 0)
                    uu += 1
                    tt = 2 * uu + 1
                    tδ = tt * (uu % 2 == 0 ? 1 : -1) * (s1^(F * tt^2))
                    tnew += tδ
                end

                l = 1 + (s1^(-F)) * tnew
            end

            total_time += abs(log(s1)) / λ
            dir = start_pos + dir * radius

            if (dir + ϵ) > Aupper
                rt = total_time + τ
                choice = 1
                return (; choice, rt)
            elseif (dir - ϵ) < Alower
                rt = total_time + τ
                choice = 0
                return (; choice, rt)
            else
                start_pos = dir
                radius = min(abs(Aupper - start_pos), (abs(Alower - start_pos)))
            end
        end
    end

# end ================== end of code from SequentialSamplingModels.jl ==================

function subject_to_idx(df)
    d = Dict(s => i for (i, s) in enumerate(unique(df.subject)))
    df.S = [d[s] for s in df.subject]
    return df
end 

function prepare_data_DDM(df; subsample = false)
    if subsample 
        println("Subsampling 10 subjects from each group for quicker testing")
        Random.seed!(2)
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

    df = @chain df begin 
        @mutate(G = case_when(lang_grp == "BE" => 1, lang_grp == "BS" => 2, lang_grp == "ME" => 3))
        #@filter(votstep == 1 || votstep == 9)
        #@mutate(VOTidx = case_when(votstep == 1 => 1, votstep == 9 => 2))
    end
        
    Vstats = @chain df @group_by(G) @summarize(V̄ = mean(VOT), σV = std(VOT)) @ungroup() @arrange(G)
    df = @chain df begin
        @group_by(G)
        @mutate(Vz  = (VOT .- mean(VOT)) ./ std(VOT))
        @ungroup()
        @select(subject, 
            lang_grp, 
            G, 
            votstep, 
            #VOTidx, 
            VOT, 
            Vz, 
            choseP, 
            RT, 
            RT_from_onset)
        @arrange(G, subject, votstep)
    end;

    df = subject_to_idx(df)

    S = df.S;
    G = @chain df @group_by(S) @slice(1) @ungroup() @pull(G);
    Vidx = df.votstep;
    V = df.VOT; 
    P = df.choseP;
    R = df.RT_from_onset ./ 1000;
    # rt = df.RT ./ 1000;
    
    return S, G, Vidx, V, P, R, df, Vstats
end

@inline function get_gamma_params(μ::Real, σ::Real)
    α = (μ / σ)^2
    θ = (σ^2) / μ
    return α, θ
end

@model function DDMhier(
    S::AbstractVector{<:Integer}, # Subject index for each row of the data
    G::AbstractVector{<:Integer}, # Group index for each subject
    V::AbstractVector{<:Real}, # VOT index
    P::AbstractVector{<:Integer}, # Choice for each row of the data (/p/ = 1, /b/ = 0)
    R::AbstractVector{<:Float64}; track::Bool = false) # RT for each row of the data

    n_groups = length(unique(G)); n_subjects = length(unique(S)); n_stim = length(unique(V))
    ϵ = 1e-3
    # minsrt_sub = [quantile(R[S .== s], 0.01) for s in 1:maximum(S)] .- ϵ
    minsrt_sub = [minimum(R[S .== s]) for s in 1:maximum(S)] .- ϵ

    # region ---- **** PRIORS **** ---- 

        # region -- *** DRIFT RATE, v *** --    
            # region -- ** GROUP LEVEL ** -- 
                v_grp ~ filldist(MvNormal(fill(0.0, n_stim), 3.0^2 * I), n_groups)
                σv ~ truncated(Normal(0, 2); lower=0.0)
            # end
                # region -- * SUBJECT LEVEL * -- 
                    v_sub_z ~ filldist(MvNormal(zeros(n_stim), I), n_subjects) 
                    v_sub := v_grp[:, G] .+ σv .* v_sub_z
                # end
        # end

        # region -- *** NON-DECISION TIME, τ *** --
            # region -- ** GROUP LEVEL ** --
                logitτ_grp ~ MvNormal(fill(0.0, n_groups), I)

                # logitτ0 ~ Normal(0.0, 1.0) 
                σlogitτ ~ truncated(Normal(0, 1.0); lower=0.0)
            # end
                # region -- * SUBJECT LEVEL * --
                    logitτ_sub_z ~ MvNormal(zeros(n_subjects), I)
                    # τ_sub := minsrt_sub .* logistic.(logitτ0 .+ σlogitτ .* logitτ_sub_z)

                    τ_sub := minsrt_sub .* logistic.(logitτ_grp[G] .+ σlogitτ .* logitτ_sub_z)
                # end
        # end

        # region -- *** BOUNDARY SEPARATION, a *** --
            # region -- ** GROUP LEVEL ** --
                a_grp ~ MvNormal(fill(log(1.5), n_groups), .75^2 * I) 

                # a0 ~ Normal(log(1.5),.75) 
                σa ~ truncated(Normal(0, .1); lower=0.0)
            # end
                # region -- * SUBJECT LEVEL * --
                    a_sub_z ~ MvNormal(zeros(n_subjects), I)
                    # a_sub := softplus.((a0 .+ σa .* a_sub_z), 6.5)

                    a_sub := softplus.((a_grp[G] .+ σa .* a_sub_z), 6.5)
                # end
        # end

        # # region -- *** INITIAL BIAS, z *** -- 
        #     # region -- ** GROUP LEVEL ** -- 
        #         # logitz0 ~ Normal(0, .5)
        #         # z0 := logistic(logitz0)
        #         z0 ~ Beta(2, 2)
        #         logitz0 = logit(z0)
        #         σlogitz ~ truncated(Normal(0, .5); lower=0.0)
        #     # end
        #         # region -- * SUBJECT LEVEL * -- 
        #             logitz_sub_z ~ MvNormal(zeros(n_subjects), I)
        #             z_sub := logistic.(logitz0 .+ σlogitz .* logitz_sub_z)
        #         # end
        # # end
    # end
        
    # region ---- **** LIKELIHOOD **** ---- #
        # When `track = true`, we also generate a predictive (choice, rt) for every trial.
        preds = track ? Vector{NamedTuple{(:choice, :rt), Tuple{Int, Float64}}}(undef, length(S)) : nothing

        for (i, s) in enumerate(S)
            v = v_sub[V[i], s]
            t = τ_sub[s]
            a = a_sub[s] # a = max(a, .1)
            z = .5 #z_sub[s]
            choice = P[i]
            rt = R[i]

            d = DDM(; ν = v, α = a, τ = t, z = z, check_args = false)
            ll = logpdf(d, choice, rt)
            Turing.@addlogprob! ll

            if track
                preds[i] = rand(d)
            end
        end

        if track
            return preds
        else
            return nothing
        end
    # end
end

votvals = collect(range(-20.0, 40.0, length=9))

@model function catDDM(
    S::AbstractVector{<:Integer}, # Subject index for each row of the data
    G::AbstractVector{<:Integer}, # Group index for each subject
    V::AbstractVector{<:Real}, # VOT index
    P::AbstractVector{<:Integer}, # Choice for each row of the data (/p/ = 1, /b/ = 0)
    R::AbstractVector{<:Float64}; track::Bool = false) # RT for each row of the data

    n_groups = length(unique(G)); n_subjects = length(unique(S)); n_stim = length(unique(V))
    ϵ = 1e-6
    # minsrt_sub = [quantile(R[S .== s], 0.01) for s in 1:maximum(S)] .- ϵ
    minsrt_sub = [minimum(R[S .== s]) for s in 1:maximum(S)] .- ϵ

    # region ---- **** CATEGORY PRIORS **** ---- 
        # region -- *** MIXING WEIGHT *** --
            # region -- ** GROUP LEVEL ** -- 
                w_grp ~ filldist(Beta(10,1), n_groups)
                logitw_grp := logit.(w_grp)
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τw_sub ~ truncated(Normal(0,1); lower=0) 
                    logitw_sub_z ~ MvNormal(zeros(n_subjects), I)
                    logitw_sub := logitw_grp[G] .+ τw_sub .* logitw_sub_z  
                    w_sub := logistic.(logitw_sub)
                # end
        # end

        # region -- *** CATEGORY MEANS *** -- 
            μₑ0 ~ MvNormal([0.0, 40.0], 5^2 * I)
            logΔμₛ0 ~ MvNormal(fill(log(40), 2), (.125^2) * I)
            Δbₛ0 := exp(logΔμₛ0[1]); Δpₛ0 := exp(logΔμₛ0[2]);

            bₑ0 := μₑ0[1]; pₑ0 := μₑ0[2];
            bₛ0 := bₑ0 - Δbₛ0; pₛ0 := pₑ0 - Δpₛ0;
            # region -- ** GROUP LEVEL ** -- 
                τμ_grp ~ truncated(Normal(0, 2.5); lower=0) 
                μ_grp_z ~ filldist(MvNormal(zeros(4), I), n_groups)
                μ_grp = [bₑ0, pₑ0, bₛ0, pₛ0] .+ τμ_grp .* μ_grp_z
                
                bₑ_grp := μ_grp[1, :]; pₑ_grp := μ_grp[2, :]; 
                bₛ_grp := μ_grp[3, :]; pₛ_grp := μ_grp[4, :]; 
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τμ_sub ~ truncated(Normal(0, 2.5); lower=0) 
                    μ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)
                    μ_sub = μ_grp[:, G] .+ τμ_sub .* μ_sub_z
                    
                    bₑ_sub := μ_sub[1, :]; pₑ_sub := μ_sub[2, :]; 
                    bₛ_sub := μ_sub[3, :]; pₛ_sub := μ_sub[4, :];
                # end
        # end

        # region -- *** CATEGORY SDs *** -- 
            σ0 ~ Normal(7.5, 2.5) 
            τσ_cat ~ truncated(Normal(0, 2.5); lower=0.0)
            σ_cat_z ~ MvNormal(zeros(4), I)
            σ_cat = softplus.(σ0 .+ τσ_cat .* σ_cat_z) .+ 1e-3

            σbₑ0 := σ_cat[1]; σpₑ0 := σ_cat[2]; 
            σbₛ0 := σ_cat[3]; σpₛ0 := σ_cat[4];
            # region -- ** GROUP LEVEL ** -- 
                τσ_grp ~ truncated(Normal(0, 2.5); lower=0.0)
                σ_grp_z ~ filldist(MvNormal(zeros(4), I), n_groups) 
                σ_grp = softplus.(σ_cat .+ τσ_grp .* σ_grp_z) .+ 1e-3
                
                σbₑ_grp := σ_grp[1, :]; σpₑ_grp := σ_grp[2, :]; 
                σbₛ_grp := σ_grp[3, :]; σpₛ_grp := σ_grp[4, :]
            # end
                # region -- * SUBJECT LEVEL * -- 
                    τσ_sub ~ truncated(Normal(0, 2.5); lower=0.0)
                    σ_sub_z ~ filldist(MvNormal(zeros(4), I), n_subjects)
                    σ_sub = softplus.(σ_grp[:, G] .+ τσ_sub .* σ_sub_z) .+ 1e-3

                    σbₑ_sub := σ_sub[1, :]; σpₑ_sub := σ_sub[2, :]; 
                    σbₛ_sub := σ_sub[3, :]; σpₛ_sub := σ_sub[4, :]
                # end
        # end

        logwₑ_sub = log.(w_sub)
        logwₛ_sub = log.(1 .- w_sub)
        logpdf_bₑ_sub = logwₑ_sub .+ hcat([logpdf.(Normal(a,b), votvals) for (a,b) in zip(bₑ_sub, σbₑ_sub)]...)'
        logpdf_pₑ_sub = logwₑ_sub .+ hcat([logpdf.(Normal(a,b), votvals) for (a,b) in zip(pₑ_sub, σpₑ_sub)]...)'
        logpdf_bₛ_sub = logwₛ_sub .+ hcat([logpdf.(Normal(a,b), votvals) for (a,b) in zip(bₛ_sub, σbₛ_sub)]...)'
        logpdf_pₛ_sub = logwₛ_sub .+ hcat([logpdf.(Normal(a,b), votvals) for (a,b) in zip(pₛ_sub, σpₛ_sub)]...)'
        log_b_sub = log.(exp.(logpdf_bₑ_sub) + exp.(logpdf_bₛ_sub))
        log_p_sub = log.(exp.(logpdf_pₑ_sub) + exp.(logpdf_pₛ_sub))

        prob_p_sub = logistic.(log_p_sub .- log_b_sub) .- .5
    # end

    # region ---- **** DDM PRIORS **** ---- 

        vscalar ~ Gamma(get_gamma_params(2.5, 1.0)...)

        v_sub = vscalar .* prob_p_sub

        # region -- *** NON-DECISION TIME, τ *** --
            # region -- ** GROUP LEVEL ** --
                logitτ_grp ~ MvNormal(fill(0.0, n_groups), I)

                # logitτ0 ~ Normal(0.0, 1.0) 
                σlogitτ ~ truncated(Normal(0, 1.0); lower=0.0)
            # end
                # region -- * SUBJECT LEVEL * --
                    logitτ_sub_z ~ MvNormal(zeros(n_subjects), I)
                    # τ_sub := minsrt_sub .* logistic.(logitτ0 .+ σlogitτ .* logitτ_sub_z)

                    τ_sub := minsrt_sub .* logistic.(logitτ_grp[G] .+ σlogitτ .* logitτ_sub_z)
                # end
        # end

        # region -- *** BOUNDARY SEPARATION, a *** --
            # region -- ** GROUP LEVEL ** --
                a_grp ~ MvNormal(fill(log(1.5), n_groups), .75^2 * I) 

                # a0 ~ Normal(log(1.5),.75) 
                σa ~ truncated(Normal(0, .1); lower=0.0)
            # end
                # region -- * SUBJECT LEVEL * --
                    a_sub_z ~ MvNormal(zeros(n_subjects), I)
                    # a_sub := softplus.((a0 .+ σa .* a_sub_z), 6.5)

                    a_sub := softplus.((a_grp[G] .+ σa .* a_sub_z), 6.5)
                # end
        # end
    # end
        
    # region ---- **** LIKELIHOOD **** ---- #
        # When `track = true`, we also generate a predictive (choice, rt) for every trial.
        preds = track ? Vector{NamedTuple{(:choice, :rt), Tuple{Int, Float64}}}(undef, length(S)) : nothing

        for (i, s) in enumerate(S)
            v = v_sub[s, V[i]]

            t = τ_sub[s]
            a = a_sub[s] # a = max(a, .1)
            z = .5 #z_sub[s]
            choice = P[i]
            rt = R[i]

            d = DDM(; ν = v, α = a, τ = t, z = z, check_args = false)
            ll = logpdf(d, choice, rt)
            Turing.@addlogprob! ll

            if track
                preds[i] = rand(d)
            end
        end

        if track
            return preds
        else
            return nothing
        end
    # end
end

function posterior_predictive_plot(samples, df)

    choices = hcat(vec([getfield.(samp, :choice) for samp in samples])...)
    rts = hcat(vec([getfield.(samp, :rt) for samp in samples])...)

    n_groups = length(unique(df.G))
    n_vots = length(unique(df.votstep))

    marg = fill(20Plots.mm, n_vots)
    p1 = plot(layout = (n_vots, n_groups), size = (250 * n_groups, 250 * n_vots), left_margin = [20 0 0].*Plots.mm, yguidefontsize=12);
    summarydf = DataFrame(G = Int[], votstep = Int[], hum_mean_choice = Float64[], sim_mean_choice = Float64[])
    for (ig, g) in enumerate(unique(df.G))
        for (iv, v) in enumerate(unique(df.votstep))
            idxs = (df.G .== g) .& (df.votstep .== v)
            
            cur_hum_rts = df.RT[idxs] ./ 1000.0
            cur_hum_choices = df.choseP[idxs]
            cur_hum_rts[cur_hum_choices .== 0] .*= -1.0

            cur_sim_rts = vec(rts[idxs, :])
            cur_sim_choices = vec(choices[idxs, :])
            # cur_sim_choices = -1 .* (cur_sim_choices .- 1) # to match human coding: /b/ = 0, /p/ = 1
            cur_sim_rts[cur_sim_choices .== 0] .*= -1.0

            push!(summarydf, (G = g, votstep = v, 
                hum_mean_choice = mean(cur_hum_choices), 
                sim_mean_choice = mean(cur_sim_choices))
            )

            xmin = 0.0
            xmax = maximum(df.RT) / 1000.0

            density!(p1[iv, ig], cur_hum_rts; color=:blue, xlims =(-xmax, xmax),
                label = ig == 3 && iv == 1 ? "Human" : nothing
                ) 
            
            density!(p1[iv, ig], cur_sim_rts; color=:red, 
                label = ig == 3 && iv == 1 ? "Sim" : nothing, 
                ylabel = ig == 1 ? "$(v)" : nothing,
                xlabel = iv == 9 ? "RT (s)" : nothing,
                yticks = ig == 1 ? :auto : nothing,
                title = iv == 1 ? "Group $(g)" : "" 
                )
            vline!(p1[iv, ig], [0.0]; linestyle = :dash, color = :black, label = "")
            # title!(p1[iv, ig], "Group $g, VOT $v")

            # bins = -0.5:1:1.5
            # # histogram!(p2[iv, ig], cur_hum_choices; bins=bins, color=:blue, label="Human", alpha=.5, normalize=true)
            # # histogram!(p2[iv, ig], cur_sim_choices; bins=bins, color=:red, label="Sim", alpha=.5, normalize=:pdf)
            # histogram!(p2[iv, ig],
            #     cur_hum_choices;
            #     bins         = -0.75:1:1.25,
            #     normalize    = :pdf,
            #     bar_width    = .5, 
            #     alpha        = 0.5,
            #     label        = "Human",
            #     color        = :blue)

            # histogram!(p2[iv, ig],
            #         cur_sim_choices;
            #         bins         = -0.25:1:1.75,
            #         normalize    = :pdf,
            #         bar_width    = .5, 
            #         alpha        = 0.5,
            #         label        = "Sim",
            #         color        = :red)

            # plot!(p2[iv, ig];
            #     xticks = ([0, 1], ["B", "P"]),
            #     xlim   = (-0.5, 1.5),
            #     xlabel = "Choice",
            # )
            # title!(p2[iv, ig], "Group $g, VOT $v")

        end
    end

    group_map = Dict(1 => "BE", 2 => "BS", 3 => "ME")
    color_map = Dict("BE" => :red, "BS" => :green, "ME" => :blue)
    summarydf.group = [group_map[g] for g in summarydf.G]

    p2 = @df summarydf plot(
        :votstep, 
        # [:hum_mean_choice, -1 .* (:sim_mean_choice .- 1)], 
        [:hum_mean_choice, :sim_mean_choice], 
        group = :group, 
        color=:G, 
        palette = [color_map[l] for l in ("BE","BS","ME")],  # order
        xlabel = "VOT", 
        ylabel = "P(/p/)", 
        ylims = (0,1),
        xticks = (1:9, votvals),
        linestyle = repeat([:solid :dash], 3), 
        linewidth = 2,
        labels = ["Hum BE" "Sim BE" "Hum BS" "Sim BS" "Hum ME" "Sim ME"]
    )

    return p1, p2
end

# ================== END MODULE ==================
end