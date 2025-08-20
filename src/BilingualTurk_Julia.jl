module BilingualTurk_Julia

    using Reexport

    project_root = dirname(Base.active_project())
    
    include(joinpath(project_root, "ModelSpecs", "BayesianModelFuncs.jl")); @reexport using .BayesianModelFuncs;
    include(joinpath(project_root, "ModelSpecs", "MouseModelFuncs.jl")); @reexport using .MouseModelFuncs;
    include(joinpath(project_root, "ModelSpecs", "StateSpaceFuncs.jl")); @reexport using .StateSpaceFuncs;
end