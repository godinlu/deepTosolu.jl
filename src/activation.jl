mutable struct Activation <: Layer
    func_activation::Function
    func_activation_prime::Function
    input
end

Activation(func_activation::Function,func_activation_prime::Function) = Activation(
    func_activation,
    func_activation_prime,
    nothing)

function forward(activation::Activation,input)
    activation.input = input
    return activation.func_activation(input)
end

function backward(activation::Activation,output_gradient,learning_rate::Float64)
    return output_gradient * activation.func_activation_prime(activation.input)
end
