using Random  # Pour la génération de nombres aléatoires


mutable struct Dense <: Layer
    input_size::Int
    output_size::Int
    weights
    bias
    input

    # Constructeur de la classe
    function Dense(input_size::Int, output_size::Int)
        new(
            input_size, 
            output_size, 
            rand(output_size, input_size), 
            rand(output_size),
            nothing
        )
    end
end

function forward(dense::Dense, input)
    dense.input = input
    return dense.weights * input + dense.bias
end

function backward(dense::Dense, output_gradient, learning_rate::float)
    weight_gradient = output_gradient * transpose(dense.input)
    input_gradient = transpose(dense.weights) * output_gradient
    dense.weights = dense.weights - learning_rate * weight_gradient
    dense.bias = dense.bias - learning_rate * output_gradient
    return input_gradient
end

dense = Dense(10,1)
output = forward(dense, rand(10))
println(output)

