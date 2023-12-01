module deepTosolu

# Write your package code here.
export Dense,forward,backward,Tanh ,mse, mse_prime,Sigmoid, Model, compile, forwardPropagation, backwardPropagation, fit, predict, evaluate
export binaryCrossEntropy, binaryCrossEntropyPrime, mse, msePrime,Convolutional, Flatten
#include("functions.Jl")
using Statistics
using Random
using ProgressBars
using LinearAlgebra
using CategoricalArrays
using Images
using DSP

include("layer.jl")
include("dense.jl")
include("convolutional.jl")
include("flatten.jl")
include("activation.jl")
include("activations.jl")
include("losses.jl")
include("model.jl")


end
