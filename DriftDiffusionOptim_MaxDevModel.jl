## TO DO: Make a MOG for the drift diffusion model surrogate, instead of the PDF approach
## - try using landscape model to generate simulated trajectories   

include("/Users/jfalanda/Documents/Projects/Bilingual_Turk/BilingualTurk_Julia/src/DriftDiffusionOptimFuncs.jl")

# @load "../Exp2(lab)_forPub/Data/simKDE.jld2" pdfs;
# pdf_itp = linear_interpolation((Arange, noiserange, md_grid), pdfs);

#@load "../Exp2(lab)_forPub/Data/simGMM_itp.jld2" gmm_itp;

#@load "../Exp2(lab)_forPub/Data/simGamma_itp.jld2" gamma_itp;

@load "../Exp2(lab)_forPub/Data/simLogNorm_itp.jld2" lognorm_itp;

paramDict₁ = OrderedDict(
    :b_eng_mean => -15.0:15.0,
    :p_eng_mean => 25.0:65.0,
    :stdev => 2.0:10.0,
    #:noise => 0.5:.1:4.0,
    :SE_scalar => .75:.1:10.0,
    #:SE_offset => -0.5:.1:0.5,
    :A_scalar => 0.1:.1:2.0,
)

paramDict₂ = OrderedDict(
    :eng_act => 0.0:.1:1.0,
    :b_eng_mean => -15.0:15.0,
    :p_eng_mean => 25.0:55.0,
    :b_spn_mean => -55.0:-25.0,
    :p_spn_mean => -15.0:15.0,
    :stdev => 5.0:15.0,
    :noise => 0.2:.1:1.0,
    #:SE_scalar => .1:.1:2.0,
    :SE_offset => -0.5:.1:0.5,
    :A_scalar => 0.1:.1:1.0,
)

results = DataFrame(
    subj = Int64[],
    lang = String[],
    LogLik = Float64[],
    #convergence = Bool[],
    eng_activation = Float64[],
    spn_activation = Float64[],
    b_eng_attractor = Float64[],
    p_eng_attractor = Float64[],
    b_spn_attractor = Float64[],
    p_spn_attractor = Float64[],
    stdev = Float64[],
    noise = Float64[],
    #SE_scalar = Float64[],
    SE_offset = Float64[],
    A_scalar = Float64[],
    prob_P = Vector{Float64}[],
    # MD = Vector{Float64}[],
    SampEn = Vector{Float64}[]
)
subj=0
lang=0
Random.seed!(1234)

# for subj in unique(rawdata.subject) #
#     # subj = unique(rawdata.subject)[3]
#     curdata = rawdata[(rawdata.subject .== subj), [:subject, :trial,:step, :language, :z_MD_above, :MD_above, :MAD,:sample_entropy, :PropP]]
#     println(subj)
for lang in unique(rawdata.language) #
    #lang = unique(rawdata.language)[3]
    curdata = rawdata[(rawdata.language .== lang), [:subject, :trial,:step, :language, :z_MD_above, :MD_above, :MAD, :sample_entropy, :PropP]]
    println(lang)

    #g(params) = optimFunc_MSE(curdata, params, 0.001, 0.0)
    g(params) = optimFunc_LogLik(curdata, params, 
        0.0, #0.1, ## Sigmoid penalty
        0.0 #1.0 ## Range penalty
        )

    bestFitParams = []
    bestFitLogLik = []
    for (i, paramDict) in enumerate([
            #paramDict₁, 
            paramDict₂])
        #paramDict = paramDict₂
        global lower = [first(value) for value in values(paramDict)]
        global upper = [last(value) for value in values(paramDict)]
        
        fittedParams = []
        LogLiks = []
        for restart in 2:9
            #initParams = [rand(Uniform(first(range), last(range))) for range in values(paramDict)]

            initParams = [range(.0,1,length=10)[restart], 0.0, 40.0, -40.0, 0.0, 7.0, .5, -.1, .2]
            #initParams = [mean(range) for range in values(paramDict)]

            initParams = scale_params(initParams, lower, upper)
            lower_ = repeat([0.0], length(initParams))
            upper_ = repeat([1.0], length(initParams))

            #initParams_sc = scale_params(initParams, lower, upper)
            res = optimize(g, lower_, upper_, initParams, Fminbox(BFGS(linesearch = LineSearches.BackTracking())), #Fminbox(NelderMead()),#
                Optim.Options( 
                    #show_trace = true, #store_trace = true,
                    iterations=100, outer_iterations=100
                ))
            convergence = Optim.converged(res)
            if convergence == true
                push!(fittedParams, unscale_params(Optim.minimizer(res), lower, upper))
                push!(LogLiks, -Optim.minimum(res))
            end
        end
        push!(bestFitParams, fittedParams[argmax(LogLiks)])
        push!(bestFitLogLik, LogLiks[argmax(LogLiks)])
    end
    BICs = [ -2 * bestFitLogLik[i] + length(bestFitParams[i]) * log(size(curdata, 1)) for i in 1:length(bestFitParams)]
    bestFitParams = bestFitParams[argmin(BICs)]
    bestFitLogLik = bestFitLogLik[argmin(BICs)]

    # if length(bestFitParams) >= 7
    #     prob_P, A = catModel_2sets(bestFitParams)
    #     noise = bestFitParams[7]
    #     SE_scalar = 1.0 #bestFitParams[7]
    #     #SE_offset = bestFitParams[9]
    #     A_scalar = bestFitParams[8]
    #     A .*= A_scalar
    # else
    #     prob_P, A = catModel_1set(bestFitParams)
    #     noise = bestFitParams[4]
    #     SE_scalar = bestFitParams[5]
    #     SE_offset = bestFitParams[6]
    #     A_scalar = bestFitParams[7]
    #     A .*= A_scalar
    # end

    prob_P, A = catModel_2sets(bestFitParams)
    noise = bestFitParams[7]
    SE_scalar = 1.0 #bestFitParams[7]
    SE_offset = bestFitParams[8]
    A_scalar = bestFitParams[9]
    A .*= A_scalar
    
    #MD_means = getSimMDMeans(noise, MD_scalar, MD_offset, A)
    SE_means = getSimSEMeans(noise, A, SE_offset)

    #combinedPlot = makePlotMD(curdata, MD_means, prob_P)
    combinedPlot = makePlotSE(curdata, SE_means, prob_P)

    if subj == 0
        savefig(combinedPlot, "../Exp2(lab)_forPub/Plots/Subj_plots/SEsimFits/LogLik/" * string(lang) * ".png")
    else
        savefig(combinedPlot, "../Exp2(lab)_forPub/Plots/Subj_plots/SEsimFits/LogLik/" * string(subj) * ".png")
    end

    eng_activation = length(bestFitParams) >= 7 ? bestFitParams[1] : 1.0
    spn_activation = 1 - eng_activation
    b_eng_attractor = length(bestFitParams) >= 7 ? bestFitParams[2] : bestFitParams[1]
    p_eng_attractor = length(bestFitParams) >= 7 ? bestFitParams[3] : bestFitParams[2]
    b_spn_attractor = length(bestFitParams) >= 7 ? bestFitParams[4] : NaN
    p_spn_attractor = length(bestFitParams) >= 7 ? bestFitParams[5] : NaN
    stdev = length(bestFitParams) >= 7 ? bestFitParams[6] : bestFitParams[3]

    out = DataFrame(
        subj = subj,
        lang = curdata.language[1],
        LogLik = bestFitLogLik,
        #convergence = convergence,
        eng_activation = eng_activation,
        spn_activation = spn_activation,
        b_eng_attractor = b_eng_attractor,
        p_eng_attractor = p_eng_attractor,
        b_spn_attractor = b_spn_attractor,
        p_spn_attractor = p_spn_attractor,
        stdev = stdev,
        noise = noise, 
        #SE_scalar = SE_scalar,
        SE_offset = SE_offset,
        A_scalar = A_scalar,
        prob_P = [prob_P],
        #MD = [MD_means]
        SampEn = [SE_means]
    )

    append!(results, out)

    if subj==0 && lang == unique(rawdata.language)[end]
        CSV.write("../Exp2(lab)_forPub/Data/MDbestfits_LogLik_lang.csv", results)
        println("Saving data")

    elseif lang ==0 && subj == unique(rawdata.subject)[end]
        CSV.write("../Exp2(lab)_forPub/Data/MDbestfits_LogLik.csv", results)
        println("Saving data")
    end
end



#rmprocs()


