using DataStructures, Printf,CSV, DataFrames, Optim, Distributed, Distributions, Turing, StatsFuns,ReverseDiff, MCMCChains, Zygote, Bijectors, JLD2 #EntropyHub #there was a conflict that led EntropyHub to downgrade several packages, including Turing
#SlurmClusterManager, 
using Random, StatsBase, Plots, Interpolations, KernelDensity, StatsPlots, JLD2, LineSearches,LsqFit, GaussianMixtures #ExpectationMaximization, ,
pyplot()

## GLOBAL VARIABLES
stimContinuum = collect(range(-20, 40, length=9));

rawdata = DataFrame(CSV.File("../Exp2(lab)_forPub/Data/data.csv"))[:,:];

Arange = 0.0:.1:.5;
noiserange = 0.1:.1:1.0;
md_grid =  -1.0:.005:7.0; # I set max to 7 because the max MD value in the dataset is ~3.0, so given the smallest scalar (.5) and offset (.5), the grid will still be larger 

#= DDM update rule: ΔZ = A*Δt + σ*sqrt(Δt)*ϵₜ
    Where Z is the decision variable, 
    A is the drift rate (constant evidence accumulation), 
    σ is the magnitude of the noise (standard deviation of fluctuations)
    sqrt(Δt) is the time scaling factor. Taking square root ensures that the noise term scales properly with time step size
    and ϵₜ is a random draw from a standard normal distribution
=#

## DDM FIXED PARAMETERS
pos_tl = (-1.0, 1.0)  # Top-left at (-1, 1)
pos_tr = (1.0, 1.0)   # Top-right at (1, 1)
Δt = 0.01  # Time step size in seconds (same as sampling rate in experiment)
s = 1.0 # scaling factor for the influence of Z on force
γ = 1.5 # damping term 
c = 5.0
k = 5.0 # stiffness constant
m = 1.0

function ddm_update(A::Float64, σ::Float64)
    return A*Δt + σ*sqrt(Δt)*randn()
end

# Function to update 2D position and velocity of the mouse cursor
function cursor_update(pos::Tuple{Float64, Float64}, vel::Tuple{Float64, Float64}, Z::Float64, s::Float64, γ::Float64)
    # Unpack the current position and velocity
    x, y = pos
    vx, vy = vel

    # Compute the weighting for the top-left and top-right forces based on Z(t)
    weight_tl = (1 - Z) / 2  # Weight for top-left force
    weight_tr = (1 + Z) / 2  # Weight for top-right force

    # Compute the direction and distance toward each target (top-left and top-right)
    dx_tl = pos_tl[1] - x
    dy_tl = pos_tl[2] - y
    dist_tl = sqrt(dx_tl^2 + dy_tl^2)

    dx_tr = pos_tr[1] - x
    dy_tr = pos_tr[2] - y
    dist_tr = sqrt(dx_tr^2 + dy_tr^2)

    # Calculate the unit vectors toward each target
    ux_tl = dist_tl > 0 ? dx_tl / dist_tl : 0.0
    uy_tl = dist_tl > 0 ? dy_tl / dist_tl : 0.0

    ux_tr = dist_tr > 0 ? dx_tr / dist_tr : 0.0
    uy_tr = dist_tr > 0 ? dy_tr / dist_tr : 0.0

    # Compute the total force by combining the forces toward top-left and top-right
    fx = s * (weight_tl * ux_tl + weight_tr * ux_tr)
    fy = s * (weight_tl * uy_tl + weight_tr * uy_tr)

    # Compute acceleration (force directly gives acceleration since mass = 1)
    ax = fx
    ay = fy

    # Update the velocities with damping
    vx += ax * Δt - γ * vx * Δt
    vy += ay * Δt - γ * vy * Δt

    # Update the positions
    x += vx * Δt
    y += vy * Δt
    if x >= 1.65
        x = 1.65
        vx = 0.0
    end
    if x <= -1.65
        x = -1.65
        vx = 0.0
    end
    if y >= 1.37
        y = 1.37
        vy = 0.0
    end
    if y <= -.076
        y = -.076
        vy = 0.0
    end

    # Return the new position and velocity as tuples
    return (x, y), (vx, vy)
end

function cursor_update2(pos::Tuple{Float64, Float64}, vel::Tuple{Float64, Float64}, Z::Float64, s::Float64, γ::Float64)
    # Unpack the current position and velocity
    x, y = pos
    vx, vy = vel

    # Compute the weighting for the top-left and top-right forces based on Z(t)
    weight_tl = (1 - Z) / 2  # Weight for top-left force
    weight_tr = (1 + Z) / 2  # Weight for top-right force

    # Compute the direction and distance toward each target (top-left and top-right)
    dx_tl = pos_tl[1] - x
    dy_tl = pos_tl[2] - y
    dist_tl = sqrt(dx_tl^2 + dy_tl^2)

    dx_tr = pos_tr[1] - x
    dy_tr = pos_tr[2] - y
    dist_tr = sqrt(dx_tr^2 + dy_tr^2)

    ax_tl = -2 * sqrt(k) * vx - k * (x - pos_tl[1])
    ax_tr = -2 * sqrt(k) * vx - k * (x - pos_tr[1])
    ax = (ax_tl * weight_tl + ax_tr * weight_tr)

    ay_tl = -2 * sqrt(k) * vy - k * (y - pos_tl[2])
    ay_tr = -2 * sqrt(k) * vy - k * (y - pos_tr[2])
    ay = (ay_tl * weight_tl + ay_tr * weight_tr)

    # Update the velocities with damping
    vx += ax * Δt 
    vy += ay * Δt 

    # β = .5
    # vx = β*vx + (1-β)*(ax * Δt)
    # vy = β*vy + (1-β)*(ay * Δt)

    # Update the positions
    x += vx * Δt
    y += vy * Δt
    if x >= 1.65
        x = 1.65
        vx = 0.0
    end
    if x <= -1.65
        x = -1.65
        vx = 0.0
    end
    if y >= 1.37
        y = 1.37
        vy = 0.0
    end
    if y <= -.076
        y = -.076
        vy = 0.0
    end

    # Return the new position and velocity as tuples
    return (x, y), (vx, vy)
end

function cursor_update3(pos::Tuple{Float64, Float64}, vel::Tuple{Float64, Float64}, Z::Float64, s::Float64, γ::Float64)
    # Unpack the current position and velocity
    x, y = pos
    vx, vy = vel

    # Compute the weighting for the top-left and top-right forces based on Z(t)
    weight_tl = (1 - Z) / 2  # Weight for top-left force
    weight_tr = (1 + Z) / 2  # Weight for top-right force

    # if Z > .0
    #     weight_tl = 0.0
    #     weight_tr = 1.0
    # elseif Z < 0.0
    #     weight_tl = 1.0
    #     weight_tr = 0.0
    # else
    #     weight_tl = 1.0
    #     weight_tr = 1.0
    # end

    # Compute the direction and distance toward each target (top-left and top-right)
    dx_tl = pos_tl[1] - x
    dy_tl = pos_tl[2] - y
    dist_tl = sqrt(dx_tl^2 + dy_tl^2)

    dx_tr = pos_tr[1] - x
    dy_tr = pos_tr[2] - y
    dist_tr = sqrt(dx_tr^2 + dy_tr^2)

    ax_tl = (-c*vx - k*(x - pos_tl[1]))/m
    ax_tr = (-c*vx - k*(x - pos_tr[1]))/m
    ax = (ax_tl * weight_tl + ax_tr * weight_tr)

    ay_tl = (-c*vy - k*(y - pos_tl[2]))/m
    ay_tr = (-c*vy - k*(y - pos_tr[2]))/m
    ay = (ay_tl * weight_tl + ay_tr * weight_tr)

    # Update the velocities with damping
    vx += ax * Δt 
    vy += ay * Δt 

    # β = .5
    # vx = β*vx + (1-β)*(ax * Δt)
    # vy = β*vy + (1-β)*(ay * Δt)

    # Update the positions
    x += vx * Δt
    y += vy * Δt
    if x >= 1.65
        x = 1.65
        vx = 0.0
    end
    if x <= -1.65
        x = -1.65
        vx = 0.0
    end
    if y >= 1.37
        y = 1.37
        vy = 0.0
    end
    if y <= -.076
        y = -.076
        vy = 0.0
    end

    # Return the new position and velocity as tuples
    return (x, y), (vx, vy)
end

function cursor_update4(pos::Tuple{Float64, Float64}, c, τ)
    # Unpack the current position and velocity
    x, y = pos

    x += -(x^3 - x + c * y) * Δt / τ
    y += -(y^2 - y + c * x) * Δt / τ
    
    return (x, y), (0.0,0.0)
end

function simulate_trial(A::Float64, σ::Float64)

    # Initial cursor position and velocity
    pos = (0.0, 0.0)  # Starting at the center
    vel = (0.0, 0.0)  # Initially at rest

    # Initialize the decision variable Z
    Z = 0.0

    # Initialize empty vectors to store the position and velocity over time
    pos_vec = [pos]
    vel_vec = [vel]

    # Initialize empty vector to store Z over time
    Z_vec = [Z]
    
    # Compute the direction and distance toward each target (top-left and top-right)
    x, y = pos
    dx_tl = pos_tl[1] - x
    dy_tl = pos_tl[2] - y

    dx_tr = pos_tr[1] - x
    dy_tr = pos_tr[2] - y

    # distance thresholds below were determined because images were 300x300, centered at (±778, 1046) on a 2560x1440 screen.
    # So the distance from center to edge was 150 px, and 150/778 = 0.19, and 150/1046 = 0.14
    while !((abs(dx_tl) < 0.19 && abs(dy_tl) < 0.14) || (abs(dx_tr) < .2 && abs(dy_tr) < .14))
        
        # the stimulus is triggered when the cursor crosses an invisible barrier at y = 100px, which is 100/1046 = .0956 in the standardized space
        if y >= .0956
            # Compute update to the decision variable Z
            Z += ddm_update(A, σ)
        else
            Z += ddm_update(0.0, 0.0)
        end

        #clamp Z to [-1, 1]
        if Z > 1.0
            Z = 1.0
        elseif Z < -1.0
            Z = -1.0
        end

        # Update the position and velocity
        # pos, vel = cursor_update(pos, vel, Z, s, γ)
        pos, vel = cursor_update4(pos, .1, 15.0)

        #store position and velocity
        push!(pos_vec, pos)
        push!(vel_vec, vel)

        #store Z
        push!(Z_vec, Z)

        # Update the distance to the response options
        x,y = pos
        dx_tl = pos_tl[1] - x
        dy_tl = pos_tl[2] - y

        dx_tr = pos_tr[1] - x
        dy_tr = pos_tr[2] - y
    end

    return pos_vec, vel_vec, Z_vec
end

# Simulate one trial
function plotOneTrial(pos_vec, Z_vec) 

    # PLOTTING
    x_positions = [pos[1] for pos in pos_vec]
    y_positions = [pos[2] for pos in pos_vec]

    DDM_plot=Plots.plot(
        collect(1:length(Z_vec)).*Δt, Z_vec, 
        ylims=(-1.0, 1.0), 
        legend=false,
        title="Drift Diffusion Model",
        size=(800,400)
    );
    hline!(DDM_plot, [0], linestyle=:dot, color=:black);

    cursor_plot=Plots.plot(
        x_positions, y_positions,
        seriestype=:scatter, 
        xlims=(-1.65, 1.65), 
        ylims=(0.0, 1.4), 
        legend=false, 
        aspect_ratio=:equal,
        title="Simulated Cursor Trajectory",
        size=(800,400)
    );
    # Add unfilled boxes to the cursor_plot
    plot!(
        cursor_plot,
        [-1.2, -0.8, -0.8, -1.2, -1.2], [0.85, 0.85, 1.15, 1.15, 0.85],
        seriestype=:shape,
        fillalpha=0,
        linecolor=:black,
        label=""
    );
    plot!(
        cursor_plot,
        [0.8, 1.2, 1.2, 0.8, 0.8], [0.85, 0.85, 1.15, 1.15, 0.85],
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

# # m=1.0
# # γ = 1.5 # damping term 
# # k = 15.0
# pos_vec, vel_vec, Z_vec = simulate_trial(
#     .3, # A
#     .1 # σ (noise magnitude)
#     );
# combined_plot = plotOneTrial(pos_vec, Z_vec)
# maximumDeviation(pos_vec)
# sampleEntropy(pos_vec)
# # plot([vec[1] for vec in vel_vec])

function maximumDeviation(pos_vec)
 
    y1 = pos_vec[1][2]
    y2 = pos_vec[end][2]
    x1 = pos_vec[1][1]
    x2 = pos_vec[end][1]

    deviations = [abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / sqrt((y2-y1)^2 + (x2-x1)^2) for (x,y) in pos_vec]

    return maximum(deviations)
end

function getMD(A, σ) 
    pos_vec, vel_vec, Z_vec =simulate_trial(A, σ)
    MD = maximumDeviation(pos_vec)
    return MD
end

function sampleEntropy(pos_vec)
    xs = [pos[1] for pos in pos_vec]
    itp=interpolate(xs, BSpline(Linear()))
    xs=itp(range(1, length(xs), length=101))
    diffs = diff(xs)
    #std(diffs)
    SampEn_ = SampEn(diffs; m=3, tau=1, r = .007) # .007 was chosen as threshold based on .2*SD(diff(x)) in the human data. The SD is across the whole dataset
    SampEn_ = SampEn_[1][4]
end

function getSampEn(A, σ) 
    #σ = 2.0
    #A = .1
    pos_vec, vel_vec, Z_vec = simulate_trial(A, σ)
    xs = [pos[1] for pos in pos_vec]
    itp=interpolate(xs, BSpline(Linear()))
    xs=itp(range(1, length(xs), length=101))
    diffs = diff(xs)
    #std(diffs)
    SampEn_ = SampEn(diffs; m=3, tau=1, r = .007) # .007 was chosen as threshold based on .2*SD(diff(x)) in the human data. The SD is across the whole dataset
    SampEn_ = SampEn_[1][4]
    return SampEn_
end

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


# Scale parameters
function scale_params(curparams, p_min, p_max)
    return (curparams .- p_min) ./ (p_max .- p_min)
end

# Reverse scaling
function unscale_params(scaled_params, p_min, p_max)
    return scaled_params .* (p_max .- p_min) .+ p_min
end

function compute_linear_trend(min_A, max_A, K)
    return [min_A + (k - 1) / (K - 1) * (max_A - min_A) for k in 1:K]
end

function penalty_deviation_from_linear(A, λ, min_A, max_A)
    K = length(A)  # Number of steps
    A_linear = compute_linear_trend(min_A, max_A, K)
    
    # Compute squared deviations from the linear trend
    penalty = λ * sum((A .- A_linear).^2)
    return penalty
end

function sigmoid_penalty(prob_P, λₛ, slope::Float64=.5, xmid::Float64=5.0)
    xvals = collect(1.0:1.0:9.0)
    #xmid = (length(prob_P) + 1) / 2
    logistic(x) = 1 / (1 + exp(-slope*(x - xmid)))

    #return λ_s * mean((prob_P .- logistic.(xvals)).^2)
    return λₛ * (prob_P .- logistic.(xvals)).^2

    # fit_result = fit_logistic(prob_P)

    # ## With 2param
    # xmid_est = fit_result.param[1]
    # slope_est = fit_result.param[2]

    # # # With 4param
    # # floor_est = fit_result.param[1]
    # # ceiling_est = fit_result.param[2]
    # # xmid_est = fit_result.param[3]
    # # slope_est = fit_result.param[4]

    # penalty = slope_est < 1.0 ? λₛ * (1 - slope_est) : 0.0
    
    # return penalty#, ceiling_est - floor_est
end

function logistic(x, p)
    # p[1]: xmid, p[2]: slope
    return 1 ./(1 .+ exp.(-p[2].*(x .- p[1])))
end

function fit_logistic(prob_P)
    xdata = collect(1.0:9.0)       # X values

    # 2 parameter
    p0 = [5.0, 1.0]                # Initial guess for xmid, slope
    fit_result = curve_fit(logistic, xdata, prob_P, p0)

    # ## 4 parameter
    # p0 = [0.0, 1.0, 5.0, 1.0]  # Initial guesses for floor, ceiling, xmid, slope
    # fit_result = curve_fit(logistic_4param, xdata, prob_P, p0)

    # fit_result.param contains fitted [xmid, slope]
    return fit_result
end

function logistic_4param(x, p)
    # p[1]: lower bound (floor), p[2]: upper bound (ceiling),
    # p[3]: midpoint (xmid), p[4]: slope
    return p[1] .+ (p[2] .- p[1]) ./ (1 .+ exp.(-p[4].*(x .- p[3])))
end

function monotonic_penalty(values, λₘ)
    penalty = 0.0
    for i in 2:length(values)
        if values[i] < values[i-1]
            penalty += (values[i-1] - values[i])^2
        end
        #penalty -= values[i] - values[i-1]
    end
    return λₘ * penalty
end

function optimFunc_LogLik(curdata, params_, λₛ, λᵣ) 

    curparams= unscale_params(params_, lower, upper)

    # if length(curparams) >=7
    #     prob_P, A = catModel_2sets(curparams)
    #     noise = curparams[7]
    #     SE_scalar = 1.0#curparams[7]
    #     #SE_offset = curparams[9]
    #     A_scalar = curparams[8]
    #     A .*= A_scalar
    # else
    #     prob_P, A = catModel_1set(curparams)
    #     noise = curparams[4]
    #     SE_scalar = curparams[5]
    #     SE_offset = curparams[6]
    #     A_scalar = curparams[7]
    #     A .*= A_scalar
    # end

    prob_P, A = catModel_2sets(curparams)
    noise = curparams[7]
    SE_scalar = 1.0#curparams[7]
    SE_offset = curparams[8]
    A_scalar = curparams[9]
    A .*= A_scalar

    ## Approach E: Interpolating Gammas
    logliks = []
    obs_per_grp = []
    for vot in 1:9
        humdata = filter(isfinite,curdata[curdata.step .== vot,:sample_entropy])
        #density(humdata, xlims=(0.0, 1.0))

        #α, θ = gamma_itp(noise, abs.(A[vot]), 1:2)
        μ, σ = lognorm_itp(noise, abs.(A[vot]), 1:2);

        # try
        #     gamma_mod = Gamma(α*sqrt(SE_scalar), θ*sqrt(SE_scalar))
        #     push!(logliks, sum(log.(pdf(gamma_mod, humdata))))#sum(log.(pdf(gamma_mod, humdata)))
        #     push!(obs_per_grp, length(humdata))
        # catch e
        #     println("α = ", α, " ;; θ = ", θ, " ;; noise = ", noise, " ;; A = ", abs.(A[vot]), " ;; SE_scalar = ", SE_scalar)
        # end

        try
            #lognorm_mod = LogNormal(μ * SE_scalar, σ * SE_scalar)
            lognorm_mod = LogNormal(μ + SE_offset, σ)

            push!(logliks, sum(log.(pdf(lognorm_mod, humdata))))#sum(log.(pdf(gamma_mod, humdata)))
            push!(obs_per_grp, length(humdata))
        catch e
            println("μ = ", μ, " ;; σ = ", σ, " ;; noise = ", noise, " ;; A = ", abs.(A[vot]), " ;; SE_scalar = ", SE_scalar)
        end

        # scl_gamma = LocationScale(0.0, SE_scalar, gamma_mod)
        # loglik += sum(log.(pdf(scl_gamma, humdata))) .+ 1e-15
        #density(rand(gamma_mod,5000), xlims=(0.0, 1.0))

    end
    totalObs = sum(obs_per_grp)
    #loglik = mean(logliks) #.* totalObs #sum(logliks) #/ totalObs
    loglik = sum(logliks) #.* totalObs #sum(logliks) #/ totalObs


    penaltyₛ = mean(sigmoid_penalty(prob_P, λₛ, 1.0, 6.0)) .* totalObs
    #penaltyₛ = penaltyₛ < .2 ? 0.0 : penaltyₛ #.* totalObs #, midpoint_)
    #penaltyₛ = sum(sigmoid_penalty(prob_P, λₛ, 1.0) .* obs_per_grp) #/ totalObs
    # penaltyₛ, range = sigmoid_penalty(prob_P, λₛ, 1.0) #.* totalObs
    #penaltyₛ = sigmoid_penalty(prob_P, λₛ, 1.0) #.* totalObs

    penaltyᵣ = (1 - (prob_P[9] - prob_P[1])) .* λᵣ #.* totalObs
    #penaltyᵣ = (1 - range) .* λᵣ .* totalObs

    #penaltyᵣ = monotonic_penalty(prob_P, λᵣ) #.* totalObs

    return -loglik + penaltyₛ + penaltyᵣ
end

function getSimMDMeans(noise, MD_scalar, MD_offset, A)
    MD_means = Float64[]
    for vot in 1:9
        μ₁, σ₁, w₁, μ₂, σ₂ = gmm_itp(noise, abs.(A[vot]), 1:5);
        if w₁ > 1.0
            w₁ = 1.0
        elseif w₁ < 0.0
            w₁ = 0.0
        end

        # scale the parameters
        μ₁ = μ₁ .* MD_scalar .+ MD_offset
        μ₂ = μ₂ .* MD_scalar .+ MD_offset
        σ₁ = σ₁ .* abs(MD_scalar)
        σ₂ = σ₂ .* abs(MD_scalar)

        gmm_mod = MixtureModel(Normal[
            Normal(μ₁, σ₁),
            Normal(μ₂, σ₂)], [w₁, 1-w₁])

        meanval = mean(gmm_mod)

        push!(MD_means, meanval)
    end
    return MD_means
end

function makePlotMD(curdata, MD_means, prob_P)
    ## Plotting for testing
    hum_ProbP = combine(groupby(curdata, [:step]), :PropP => mean => :ProbP).ProbP
    hum_MDs = combine(groupby(curdata, [:step]), :MD_above => mean => :mean_MD).mean_MD #./ MD_scalar .- MD_offset

    p1 = plot(stimContinuum, hum_ProbP, ylims=(0,1), size=(300,300), linecolor=:black, title = "Prob(P)", label="Human", margin = 5Plots.mm);
    plot!(p1, stimContinuum, prob_P,linecolor=:red, label = "Sim");
    MSE = mean((hum_ProbP .- prob_P).^2)
    annotate!(p1, :bottomright, @sprintf("MSE: %.2e", MSE),8)

    p2 = plot(stimContinuum, hum_MDs, linecolor=:black, size=(300,300), title = "Mean(MD)", legend = false, margin=5Plots.mm);
    plot!(p2, stimContinuum, MD_means, linecolor=:red);
    MSE = mean((hum_MDs .- MD_means).^2)
    annotate!(p2, :bottomright, @sprintf("MSE: %.2e", MSE),8)
    
    combinedPlot = plot(p1,p2, layout=(2,1), size=(300,600));
    #combinedPlot
    return combinedPlot
end

function getSimSEMeans(noise, A, SE_offset)
    
    SE_means = Float64[]
    # for vot in 1:9
    #     α, θ = gamma_itp(noise, abs.(A[vot]), 1:2);
    #     gamma_mod = Gamma(α*sqrt(SE_scalar), θ*sqrt(SE_scalar))
    #     meanval = mean(gamma_mod)

    #     # scl_gamma = LocationScale(0.0, SE_scalar, gamma_mod)
    #     # meanval = mean(scl_gamma)

    #     push!(SE_means, meanval)
    # end

    for vot in 1:9
        μ, σ = lognorm_itp(noise, abs.(A[vot]), 1:2);
        
        #lognorm_mod = LogNormal(μ * SE_scalar, σ * SE_scalar)
        lognorm_mod = LogNormal(μ + SE_offset, σ)

        meanval = mean(lognorm_mod)

        push!(SE_means, meanval)
    end

    return SE_means
end

function makePlotSE(curdata, SE_means, prob_P)
    ## Plotting for testing
    hum_ProbP = combine(groupby(curdata, [:step]), :PropP => mean => :ProbP).ProbP
    filtered_df = filter(row -> isfinite(row.sample_entropy), curdata)
    hum_SEs = combine(groupby(filtered_df, :step), :sample_entropy => mean => :mean_SE).mean_SE
    
    p1 = plot(stimContinuum, hum_ProbP, ylims=(0,1), size=(300,300), linecolor=:black, title = "Prob(P)", label="Human", margin = 5Plots.mm);
    plot!(p1, stimContinuum, prob_P,linecolor=:red, label = "Sim");
    MSE = mean((hum_ProbP .- prob_P).^2)
    annotate!(p1, :bottomright, @sprintf("MSE: %.2e", MSE),8)

    p2 = plot(stimContinuum, hum_SEs, linecolor=:black, size=(300,300), title = "Mean(SampEn)", legend = false, margin=5Plots.mm);
    plot!(p2, stimContinuum, SE_means, linecolor=:red);
    MSE = mean((hum_SEs .- SE_means).^2)
    annotate!(p2, :bottomright, @sprintf("MSE: %.2e", MSE),8)
    
    combinedPlot = plot(p1,p2, layout=(2,1), size=(300,600));
    #combinedPlot
    return combinedPlot
end




