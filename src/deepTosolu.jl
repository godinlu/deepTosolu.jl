module deepTosolu

# Write your package code here.
export Dense,forward,backward,Tanh ,mse, mse_prime,Sigmoid, Model, compile, forwardPropagation, backwardPropagation, fit, predict
#include("functions.Jl")
using Statistics
using Random
using ProgressBars
using LinearAlgebra
using CategoricalArrays

include("layer.jl")
include("dense.jl")
include("activation.jl")
include("activations.jl")
include("losses.jl")
include("model.jl")


end
