

mutable struct Dense <: Layer
    input_size::Int
    output_size::Int
    weights::Matrix{Float64}
    bias::Vector{Float64}
    input::Vector{<:Number}

    # Constructeur de la classe
    function Dense(input_size::Int, output_size::Int)
        new(
            input_size, 
            output_size, 
            rand(output_size, input_size), 
            rand(output_size),
            Float64[]
        )
    end
end

function forward(dense::Dense, input::Vector{<:Number})
    dense.input = input
    return dense.weights * input + dense.bias
end

function backward(dense::Dense, output_gradient::Vector{<:Number}, learning_rate::Float64)
    weight_gradient = output_gradient * transpose(dense.input)
    input_gradient = transpose(dense.weights) * output_gradient
    dense.weights = dense.weights - learning_rate * weight_gradient
    dense.bias = dense.bias - learning_rate * output_gradient
    return input_gradient
end


