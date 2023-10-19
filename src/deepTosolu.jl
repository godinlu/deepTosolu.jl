module deepTosolu

# Write your package code here.
export Dense,forward,backward,Tanh ,mse, mse_prime,Sigmoid, Model, compile, forwardPropagation, backwardPropagation
#include("functions.Jl")
using Statistics, Random

include("layer.jl")
include("dense.jl")
include("activation.jl")
include("activations.jl")
include("losses.jl")
include("model.jl")


end
