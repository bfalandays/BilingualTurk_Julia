module MouseModel 
# ================== START MODULE ==================

export mDDM,
    syntheticLogLik,
    optimFunc,
    simTrial_getMeasures,
    getAllMeasures,
    getDevMeasures,
    getSampEn,
    simulate_trial,
    sampleEntropy,
    timeNormalize,
    plotTrial
    
using Reexport
@reexport using ..Common, Optim
using LaTeXStrings, LinearAlgebra, GaussianMixtures, Interpolations, KernelDensity, Distributed

#= DDM FIXED PARAMETERS
    # Screen setup #
    starting position was always (0,48)
    sound was played when boundary crossed y = 148
    response options were centered at (-778, 1114) and (+822, 1114) and were 300 x 300 px
=#
pos_tl::Tuple{Float64, Float64} = (-800, 1114) #(-778/1280, 1046/1440)#(-1.0, 1.0)  # Top-left at (-1, 1)
pos_tr::Tuple{Float64, Float64} = (800, 1114) #(778/1280, 1046/1440)#(1.0, 1.0)   # Top-right at (1, 1)
xmargin::Float64 = 150.0#/1280
ymargin::Float64 = 150.0#/1440
x_start::Float64, y_start::Float64 = (0.0, 48.0)
min_x::Float64, max_x::Float64 = (-1280, 1280)
min_y::Float64, max_y::Float64 = (0, 1440)
Δt::Float64 = 0.01  # Time step size in seconds (same as sampling rate in experiment)

function simulate_trial(
    A::Float64, 
    ε::Float64, 
    Z_thresh::Float64=.8, 
    β::Float64=5.0, 
    k::Float64=5.0, 
    cₖ::Float64=1.0; 
    Δt::Float64=Δt, 
    hit_x::Float64=xmargin, hit_y::Float64=ymargin, barrier_y::Float64=148.0,
    x_start::Float64=x_start, y_start::Float64=y_start, 
    min_x::Float64=min_x,max_x::Float64=max_x,
    min_y::Float64=min_y,max_y::Float64=max_y)

    max_tsteps = Int(ceil(5/Δt))

    x = x_start; y = y_start;
    vx = 0.0; vy = 0.0;
    Z = 0.0;
    Zs = Vector{Float64}(undef, max_tsteps+1);
    Zs[1] = Z;

    c = cₖ * -2 * sqrt(k)
    sΔ = sqrt(Δt)

    xs = Vector{Float64}(undef, max_tsteps+1)
    ys = Vector{Float64}(undef, max_tsteps+1)
    xs[1] = x; ys[1] = y

    tstep = 1
    stimTime = 0.0
    crossedbarrier = false
    @inbounds @fastmath while tstep < max_tsteps
        tstep += 1
        
        if y >= barrier_y && !crossedbarrier
            crossedbarrier = true
        end
                
        # DDM increment (inline)
        if crossedbarrier
            stimTime += Δt
            # input = (stimTime ≤ 1.5) ? A : 0.0
            input = A
            Z += input*Δt + ε*sΔ*randn()
        else
            Z += ε*sΔ*randn()
        end
        # Z = clamp(Z, -Z_thresh, Z_thresh)
        Zs[tstep] = Z;

        # logistic weight and accelerations
        wtr = 1.0 / (1.0 + exp(-β * Z))
        wtl = 1.0 - wtr

        ax = (c*vx - k*(x - pos_tl[1]))*wtl + (c*vx - k*(x - pos_tr[1]))*wtr
        ay = (c*vy - k*(y - pos_tl[2]))*wtl + (c*vy - k*(y - pos_tr[2]))*wtr

        vx += ax * Δt #+ .02 * sqrt(Δt) * randn()
        vy += ay * Δt #+ .02 * sqrt(Δt) * randn()
        x += vx * Δt; x = clamp(x, min_x, max_x)
        y += vy * Δt; y = clamp(y, min_y, max_y)

        xs[tstep] = x; ys[tstep] = y

        # hit test vs boxes (early exit)
        inbox = ((abs(pos_tl[1]-x) < hit_x && abs(pos_tl[2]-y) < hit_y) || (abs(pos_tr[1]-x) < hit_x && abs(pos_tr[2]-y) < hit_y))
        v = sqrt(vx^2 + vy^2)
        if (abs(Z) >= Z_thresh) && (tstep*Δt ≥ 0.25) && inbox && v < 10.0
            break
        end

    end

    if Z >= Z_thresh
        choice = 1
    else
        choice = 0
    end

    # if tstep >= max_tsteps
    #     return nothing
    # end

    return view(xs, 1:tstep), view(ys, 1:tstep), view(Zs, 1:tstep), choice
end

# Simulate one trial
function plotTrial(x, y, Z) 

    minZ = minimum([minimum(Z), -1.0]); maxZ = maximum([maximum(Z), 1.0]);
    DDM_plot=Plots.plot(
        collect(1:length(Z)).*Δt, Z, 
        ylims=(minZ, maxZ), 
        legend=false,
        title="Drift Diffusion Model",
        size=(800,400)
    );
    hline!(DDM_plot, [0], linestyle=:dot, color=:black);

    cursor_plot=Plots.plot(
        x, y,
        seriestype=:scatter, 
        xlims=(min_x, max_x), 
        ylims=(min_y, max_y), 
        legend=false, 
        aspect_ratio=:equal,
        title="Simulated Cursor Trajectory",
        size=(800,400)
    );
    # Add unfilled boxes to the cursor_plot
    plot!(
        cursor_plot,
        #[-1.2, -0.8, -0.8, -1.2, -1.2], [0.85, 0.85, 1.15, 1.15, 0.85],
        [pos_tl[1] - xmargin, pos_tl[1] + xmargin, pos_tl[1] + xmargin, pos_tl[1] - xmargin, pos_tl[1] - xmargin],
        [pos_tl[2] - ymargin, pos_tl[2] - ymargin, pos_tl[2] + ymargin, pos_tl[2] + ymargin, pos_tl[2] - ymargin],
        seriestype=:shape,
        fillalpha=0,
        linecolor=:black,
        label=""
    );
    plot!(
        cursor_plot,
        #[0.8, 1.2, 1.2, 0.8, 0.8], [0.85, 0.85, 1.15, 1.15, 0.85],
        [pos_tr[1] - xmargin, pos_tr[1] + xmargin, pos_tr[1] + xmargin, pos_tr[1] - xmargin, pos_tr[1] - xmargin],
        [pos_tr[2] - ymargin, pos_tr[2] - ymargin, pos_tr[2] + ymargin, pos_tr[2] + ymargin, pos_tr[2] - ymargin],
        seriestype=:shape,
        fillalpha=0,
        linecolor=:black,
        label=""
    );

    combined_plot=Plots.plot(
        DDM_plot, cursor_plot, 
        layout = @layout([a; b]),
        size=(800,800)
    );

    return combined_plot
    
end

function timeNormalize(x::AbstractVector{<:Real}, y::AbstractVector{<:Real})
    n = length(x)
    if n == 1
        xs = fill(float(x[1]), 101); ys = fill(float(y[1]), 101); return xs, ys
    end
    tvals = range(1, n, length=101)
    xs = Vector{Float64}(undef, 101); ys = similar(xs)
    @inbounds for (k, t) in enumerate(tvals)
        i = clamp(floor(Int, t), 1, n-1)
        α = t - i
        xs[k] = x[i] + α*(x[i+1]-x[i])
        ys[k] = y[i] + α*(y[i+1]-y[i])
    end
    return xs, ys
end

function sampEn_count_pairwise_matches(w::Vector{Vector{Float64}}, r::Float64)
    matches::Int64=0
    for i in 1:length(w)-1
        for j in i+1:length(w)
            dist = maximum(abs.(w[i].- w[j]))
            if dist <= r
                matches += 1
            end
        end
    end
    return matches
end

function sampleEntropy(ẋ::Vector{Float64}, m::Int64=3, r::Float64=0.0)

    if r == 0.0
        r = .2 * std(ẋ)
    end

    wₘ = [ẋ[i:i+m-1] for i in 1:(length(ẋ)-m)] 
    B = sampEn_count_pairwise_matches(wₘ, r)  # count matches for m-length vectors
    wₘ₊₁ = [ẋ[i:i+m] for i in 1:(length(ẋ)-m)]
    A = sampEn_count_pairwise_matches(wₘ₊₁, r)  # count matches for m+1-length vectors

    sampEn = -log(A / B)

    return sampEn
end

function getSampEn(x::AbstractVector{<:Real}, m::Int64=3, r::Float64=0.0)
 
    ẋ = diff(x)
    sampEn = sampleEntropy(ẋ, m, r)

    return sampEn
end

function getDevMeasures(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}; out::String="both")
    x1 = first(x); x2 = last(x)
    y1 = first(y); y2 = last(y)

    deviations = [abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / sqrt((y2-y1)^2 + (x2-x1)^2) for (x,y) in zip(x, y)]

    if out=="both"
        return maximum(deviations), mean(deviations)
    elseif out=="MD"
        return maximum(deviations)
    elseif out=="AD"
        return mean(deviations)
    end
end

function getAllMeasures(x::AbstractVector{<:Real}, y::AbstractVector{<:Real}, m::Int64=3, r::Float64=0.0)

    MD, AD = getDevMeasures(x, y)
 
    sampEn = getSampEn(x, m, r)

    return MD, AD, sampEn
end

function simTrial_getMeasures(A::Float64; ε::Float64=1.0, Z_thresh::Float64=2.0, β::Float64=5.0, k::Float64=59.75, cₖ::Float64=0.61, m::Int64=3, r::Float64=0.0, out::String="all")
    
    x, y, Z, choice = simulate_trial(A, ε, Z_thresh, β, k, cₖ)
    
    #x, y = timeNormalize(x, y)

    if out == "all"
        MD, AD, sampEn = getAllMeasures(x, y, m, r)
        return MD, AD, sampEn, choice
    elseif out=="devs"
        MD, AD = getDevMeasures(x, y; out="both")
        return MD, AD, choice
    elseif out == "MD" || out == "AD"
        XD = getDevMeasures(x, y; out=out)
        return XD, choice
    elseif out == "sampEn"
        sampEn = getSampEn(x, y, m, r)
        return sampEn, choice
    end
end

#= test plotting
    x, y, Z, choice = simulate_trial(
        1.0, # A
        1.0, # ε (noise magnitude)
        2.0, # Z threshold for decision
        3.25, # β (logistic slope for weight as a function of Z)
        158.0, # k 
        .51 # # cₖ
        );
    combined_plot = plotTrial(x,y,Z)
    # x, y = timeNormalize(x, y);
    MD, AD, sampEn = getAllMeasures(x,y)
=#

function catModel_2sets(curparams)
    eng_act = curparams[1] #abs(params[1])
    if eng_act > 1
        eng_act = 1
    elseif eng_act < 0
        eng_act = 0
    end
    weights = [eng_act, 1-eng_act] #./ sum([eng_act, 1-eng_act])
    b_eng_mean = curparams[2]
    p_eng_mean = curparams[3]
    b_spn_mean = curparams[4]
    p_spn_mean = curparams[5]
    stdev = curparams[6]

    b_GMM = MixtureModel(Normal[
        Normal(b_eng_mean, stdev),
        Normal(b_spn_mean, stdev)],
        weights)

    p_GMM = MixtureModel(Normal[
        Normal(p_eng_mean, stdev),
        Normal(p_spn_mean, stdev)],
        weights)

    pdf_P = pdf(p_GMM, stimContinuum)
    pdf_B = pdf(b_GMM, stimContinuum)

    prob_P = pdf_P ./ (pdf_P + pdf_B)
    
    A = (prob_P .- .5)
    #A = (2 .* prob_P) .- 1
    #A = tanh.(.2* log.(prob_P ./ (1 .- prob_P)))

    return prob_P, A
end

function catModel_1set(curparams)
    b_eng_mean = curparams[1]
    p_eng_mean = curparams[2]
    stdev = curparams[3]

    b_GMM = Normal(b_eng_mean, stdev)

    p_GMM = Normal(p_eng_mean, stdev)

    pdf_P = pdf(p_GMM, stimContinuum)
    pdf_B = pdf(b_GMM, stimContinuum)

    #A = (prob_P .- .5)
    A = (2 .* prob_P) .- 1
    #A = tanh.(.2 * log.(prob_P ./ (1 .- prob_P)))

    return prob_P, A
end

#=
    humdata = data.MD₂
    β, k, cₖ = 8.32, 443.62, 1.32
    A = 1.0
    sim_ys = pmap((trial) -> simTrial_getMeasures(A; β = β, k = k, cₖ = cₖ, out="MD"), 1:length(humdata));
    density(humdata, xlims=(minimum(humdata), maximum(humdata)))
    density!(sim_ys, xlims=(minimum(humdata), maximum(humdata)))
=#

function optimFunc(data, params) 
    β, k, cₖ = params

    humdata = data.MD₂
    B_ys_hum = humdata[data.choseP .== 0]; P_ys_hum = humdata[data.choseP .== 1];
    
    # σ, σ_τ, k, cₖ = [0.698, 0.449, 59.75, 0.61]
    A = 0.0
    simdata = pmap((trial) -> simTrial_getMeasures(A; β = β, k = k, cₖ = cₖ, out="MD"), 1:length(humdata));
    sim_ys = first.(simdata); sim_choices = last.(simdata); B_ys_sim = sim_ys[sim_choices .== 0]; P_ys_sim = sim_ys[sim_choices .== 1];

    bw = 1.06 * std(humdata) * length(humdata)^(-1/5)
    kde_fit = kde(sim_ys, boundary=(minimum(humdata)-1.0, maximum(humdata)+1.0), bandwidth=bw)
    loglik = sum(log.(pdf(kde_fit, humdata)))
    
    # gmm = MixtureModel(GMM(2,sim_ys));
    # loglik = loglikelihood(gmm, humdata)

    # d = fit(Gamma{Float64}, sim_ys)
    # loglik = loglikelihood(d, humdata)

    return -loglik
end

function syntheticLogLik(y::Float64, A::Float64; μₑ::Float64=0.698, σₑ::Float64=0.449, k::Float64=59.75, cₖ::Float64=0.61, m::Int64=3, r::Float64=0.0, lb::Float64=0.0, ub::Float64=2.0, nSims::Int64=1000, out::String="all") 
    
    sim_ys = pmap((trial) -> simTrial_getMeasures(A; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, m = m, r = r, out="AD"), 1:nSims);

    # d = kde(sim_ys, boundary=(lb, ub))
    # loglik = log(clamp(pdf(d, y), eps(), Inf))

    d = fit(Gamma{Float64}, sim_ys)
    loglik = logpdf(d, y)

    return loglik
end

function simPDF(A::Float64; μₑ::Float64=0.698, σₑ::Float64=0.449, k::Float64=59.75, cₖ::Float64=0.61, m::Int64=3, r::Float64=0.0, lb::Float64=0.0, ub::Float64=2.0, nSims::Int64=1000, out::String="all") 
    
    sim_ys = pmap((trial) -> simTrial_getMeasures(A; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, m = m, r = r, out="AD"), 1:nSims);

    # d = kde(sim_ys, boundary=(lb, ub))
    # loglik = log(clamp(pdf(d, y), eps(), Inf))

    d = fit(Gamma{Float64}, sim_ys)

    return d
end

@model function mDDM(
    S::AbstractVector{<:Real}, # S ϵ ℝᴶ ; S[j] indexes the subject for trial j
    G::AbstractVector{<:Real}, # G ϵ ℝᴶ ; G[j] indexes the language group to which subject S[j] belongs
    V::AbstractVector{<:Real}, # V ϵ ℝᴶ ; V[j] is the stimulus VOT value (scaled to -1:1) on trial j
    y::AbstractVector{<:Real}) # y ϵ ℝᴶ ; y[j, n] is the observed trajectory measure on trial j
    
    lb = minimum(y); ub = maximum(y)

    μₑ ~ truncated(Normal(); lower=0)
    σₑ ~ truncated(Normal(); lower=0)
    k ~ truncated(Cauchy(0,5); lower=0)
    cₖ ~ truncated(Normal(1,1); lower=0)
    
    β₀ ~ filldist(Normal(0,1), 3) 
    βᵥ ~ filldist(Normal(), 3)
    Aₖ ~ truncated(Normal(); lower=0)

    A = β₀[G] .+ logistic.(V .* βᵥ[G]) .* Aₖ

    dDict = Dict(Aᵢ => simPDF(Aᵢ; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, lb = lb, ub = ub, nSims=1000) for (i, Aᵢ) in enumerate(unique(A)))

    for (i,yᵢ) in enumerate(y)
        # @addlogprob! syntheticLogLik(yᵢ, A[i]; μₑ = μₑ, σₑ = σₑ, k = k, cₖ = cₖ, lb = lb, ub = ub, nSims=1000)
        @addlogprob! loglikelihood(dDict[A[i]], yᵢ)
    end
end

# ================== END MODULE ==================
end