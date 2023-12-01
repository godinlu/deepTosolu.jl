mutable struct Flatten <: Layer
    input_shape::Tuple{Int, Int, Int}
    output_shape::Tuple{Int, Int}

    function Flatten(input_shape::Tuple{Int, Int, Int}, output_shape::Tuple{Int, Int})
        new(
            input_shape,
            output_shape
        )
    end
end

function forward(obj::Flatten, input::Array{<:Number,3})
    return reshape(input, obj.output_shape...)
end

function backward(obj::Flatten, output_gradient::Matrix{<:Number}, learning_rate::Float64)
    return reshape(output_gradient, obj.input_shape...)
end