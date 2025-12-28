module Common
# ================== START MODULE ==================

export dviz, subject_to_idx, stimContinuum

using Reexport
@reexport using DataFrames, TidierData, Random, StatsBase, StatsFuns, Distributions, Turing, DynamicPPL, ReverseDiff, Mooncake, Plots, StatsPlots, CSV, JLD2
gr()

function dviz(d) # convenience function for visualizing distributions
    tmp = rand(d,10000)
    if typeof(d) <: Beta
        display(histogram(tmp, xlims=(0,1)))
    else
        display(histogram(tmp))
    end
    
    return describe(tmp)
end

function subject_to_idx(df)
    d = Dict(s => i for (i, s) in enumerate(unique(df.subject)))
    df.S = [d[s] for s in df.subject]
    return df
end 

stimContinuum = collect(range(-20, 40, length=9));

# ================== END MODULE ==================
end