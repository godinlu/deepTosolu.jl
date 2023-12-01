mutable struct Convolutional <: Layer
    depth::Int
    input_shape::Tuple{Int,Int,Int}
    input_depth::Int
    output_shape::Tuple{Int,Int,Int}
    kernels_shape::Tuple{Int,Int,Int,Int}
    kernels::Array{<:Number,4}
    biases::Array{<:Number,3}
    input::Array
    output::Array

    #constructeur de la class
    function Convolutional(
        input_shape::Tuple{Int,Int,Int},
        kernel_size::Int,
        depth::Int
    )::Convolutional
    input_depth, input_height, input_width = input_shape
    output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
    kernels_shape = (depth, input_depth, kernel_size, kernel_size)
    new(
        depth,
        input_shape,
        input_depth,
        output_shape,
        kernels_shape,
        randn(kernels_shape...),
        randn(output_shape...)
    )

    end
end

function correlate2d_valid(input, kernel)
    # Tailles de l'entrée et du noyau
    input_size = size(input)
    kernel_size = size(kernel)

    # Calcul de la taille de sortie pour le mode "valid"
    output_size = (input_size .- kernel_size) .+ 1

    # Convolution avec le mode "full" pour obtenir le résultat complet
    result_full = conv(input, kernel)

    # Extraction de la partie "valid" du résultat en fonction de la taille souhaitée
    valid_start = div.(kernel_size, 2) .+ 1
    valid_end = valid_start .+ output_size .- 1

    # Sélection de la partie "valid" du résultat
    result_valid = result_full[valid_start[1]:valid_end[1], valid_start[2]:valid_end[2]]

    return result_valid
end

function forward(convolutional::Convolutional, input::Array)
    convolutional.input = input
    convolutional.output = copy(convolutional.biases)

    for i in 1:convolutional.depth
        for j in 1:convolutional.input_depth
            convolutional.output[i,:,:] += correlate2d_valid(convolutional.input[j,:,:], convolutional.kernels[i, j,:,:])
        end
    end
    return convolutional.output 
end


function backward(convolutional::Convolutional, output_gradient, learning_rate::Float64)
    kernels_gradient = zeros(convolutional.kernels_shape)
    input_gradient = zeros(convolutional.input_shape)
    println(size(kernels_gradient))

    for i in 1:convolutional.depth
        for j in 1:convolutional.input_depth
            kernels_gradient[i,j,:,:] = correlate2d_valid(convolutional.input[j,:,:], output_gradient[i,:,:])
            input_gradient[j,:,:] = conv(output_gradient[i,:,:], convolutional.kernels[i,j,:,:])
        end
    end

    convolutional.kernels -= learning_rate * kernels_gradient
    convolutional.biases -= learning_rate * output_gradient
    return input_gradient
end