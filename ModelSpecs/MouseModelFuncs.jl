module MouseModelFuncs 
# ================== START MODULE ==================
export optimFunc,
    simTrial_getMD

using Reexport
@reexport using DataFrames, CSV, Distributed, Distributions, StatsFuns, Turing, ParetoSmooth, ReverseDiff, Plots, Random, LaTeXStrings, LinearAlgebra, GaussianMixtures, Optim, JLD2

stimContinuum = collect(range(-20, 40, length=9));

## DDM FIXED PARAMETERS
pos_tl = (-1.0, 1.0)  # Top-left at (-1, 1)
pos_tr = (1.0, 1.0)   # Top-right at (1, 1)
Δt = 0.01  # Time step size in seconds (same as sampling rate in experiment)

function ddm_update(A::Float64, σ::Float64)
    #= DDM update rule: ΔZ = A*Δt + σ*sqrt(Δt)*ϵₜ
        Where Z is the decision variable, 
        A is the drift rate (constant evidence accumulation), 
        σ is the magnitude of the noise (standard deviation of fluctuations)
        sqrt(Δt) is the time scaling factor. Taking square root ensures that the noise term scales properly with time step size
        and ϵₜ is a random draw from a standard normal distribution
    =#
    return A*Δt + σ*sqrt(Δt)*randn()
end

# damped-spring model, assuming mass = 1.0 and using critical damping
function cursor_update(pos::Tuple{Float64, Float64}, vel::Tuple{Float64, Float64}, Z::Float64, k::Float64=5.0, c_scalar::Float64=1.0)
    # Unpack the current position and velocity
    x, y = pos
    vx, vy = vel

    # Compute the weighting for the top-left and top-right forces based on Z(t)
    # weight_tl = (1 - Z) / 2  # Weight for top-left force
    # weight_tr = (1 + Z) / 2  # Weight for top-right force

    β = 5.0  # steepness
    weight_tr = 1 ./ (1 .+ exp.(-β .* Z))
    weight_tl = 1 - weight_tr

    c = c_scalar * -2 * sqrt(k)  # Damping coefficient for critical damping

    # Compute the direction and distance toward each target (top-left and top-right)
    ax_tl = c * vx - k * (x - pos_tl[1])
    ax_tr = c * vx - k * (x - pos_tr[1])
    ax = (ax_tl * weight_tl + ax_tr * weight_tr)

    ay_tl = c * vy - k * (y - pos_tl[2])
    ay_tr = c * vy - k * (y - pos_tr[2])
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

function simulate_trial(A::Float64, σ::Float64, k::Float64=5.0, c_scalar::Float64=1.0)

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
    # d_tl = sqrt(dx_tl^2 + dy_tl^2)

    dx_tr = pos_tr[1] - x
    dy_tr = pos_tr[2] - y
    # d_tr = sqrt(dx_tr^2 + dy_tr^2)

    # Images were 300x300, centered at (±778, 1046) on a 2560x1440 screen. So (778,1046) becomes (1,1)
    # The distance from center to edge was 150 px, and let's say participants don't click right on the edge--give it a 25px buffer. 125/778 = 0.16, and 125/1046 = 0.12
    while !((abs(dx_tl) < 0.06 && abs(dy_tl) < 0.047) || (abs(dx_tr) < .06 && abs(dy_tr) < .047))
        
        # the stimulus is triggered when the cursor crosses an invisible barrier at y = 100px, which is 100/1046 = .0956 in the standardized space
        if y >= .0956
            # Compute update to the decision variable Z
            Z += ddm_update(A, σ)
        else
            Z += ddm_update(0.0, σ)
        end

        #clamp Z to [-1, 1]
        if Z > 1.0
            Z = 1.0
        elseif Z < -1.0
            Z = -1.0
        end

        # Update the position and velocity
        pos, vel = cursor_update(pos, vel, Z, k, c_scalar)

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

# pos_vec, vel_vec, Z_vec = simulate_trial(
#     1.0, # A
#     .78, # σ (noise magnitude)
#     60.0, # k
#     .6
#     );
# combined_plot = plotOneTrial(pos_vec, Z_vec)
# maximumDeviation(pos_vec)

function maximumDeviation(pos_vec)
 
    y1 = pos_vec[1][2]
    y2 = pos_vec[end][2]
    x1 = pos_vec[1][1]
    x2 = pos_vec[end][1]

    deviations = [abs((y2-y1)*x - (x2-x1)*y + x2*y1 - y2*x1) / sqrt((y2-y1)^2 + (x2-x1)^2) for (x,y) in pos_vec]

    return maximum(deviations)
end

function simTrial_getMD(A::Float64, σ::Float64, σ_τ::Float64, k::Float64=5.0, c_scalar::Float64=1.0) 
    σ₂ = rand(truncated(Normal(σ,σ_τ);lower=0, upper=2.0))
    pos_vec, vel_vec, Z_vec = simulate_trial(A, σ₂, k, c_scalar)
    MD = maximumDeviation(pos_vec)
    return MD
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

# simdata = pmap((trial) -> simTrial_getMD(1.0, .9,.4, 60.0, .7), 1:5000);
# density(simdata)
# density!(df9.MD_above)

function optimFunc(data, params) 
    σ = params[1]
    σ_τ = params[2]
    k = params[3]
    c_scalar = params[4] #c_scalar is the damping coefficient
    #m = params[4]

    humdata = data.MD_above
    bw = 1.06 * std(humdata) * length(humdata)^(-1/5)

    simdata = pmap((trial) -> simTrial_getMD(1.0, σ, σ_τ, k, c_scalar), 1:length(humdata));

    # kde_fit = kde(simdata, boundary=(minimum(humdata)-1.0, maximum(humdata)+1.0), bandwidth=bw)
    # densities = pdf(kde_fit, humdata)
    # clipped = clamp.(densities, 1e-6, Inf)  # eps() ≈ 2.22e-16
    # log_lik = sum(log.(clipped))
    # log_lik = sum(log.(pdf(kde_fit, humdata)))

    # density(humdata, xlims=(minimum(humdata), maximum(humdata)))
    # density!(simdata, xlims=(minimum(humdata), maximum(humdata)))
    # y = pdf(kde_fit, 0.:.01:2.0) #collect(range(minimum(humdata), maximum(humdata), length(kde_fit.density)))
    # plot!(0.0:.01:2.0,y)
    
    gmm = MixtureModel(GMM(2,simdata));
    log_lik = loglikelihood(gmm, humdata)

    return -log_lik
end

# ================== END MODULE ==================
end