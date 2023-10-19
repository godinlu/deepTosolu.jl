mutable struct Model
    layers::Vector{Layer}
    loss_func::Function
    loss_func_prime::Function

end

Model(layers::Vector{Layer}) = Model(
    layers,
    mse,
    msePrime
    )

function setLoss(model::Model, loss_name::String)
    if loss_name == "mse"
        model.loss_func = mse
        model.loss_func_prime = msePrime
    elseif loss_name == "binaryCrossEntropy"
        model.loss_func = binaryCrossEntropy
        model.loss_func_prime = binaryCrossEntropyPrime
    else
        error("error the loss function "* loss_name *" does not exist.")
    end

end


function compile(model::Model, loss_name::String = "mse")
    setLoss(model, loss_name)
end

function forwardPropagation(model::Model, input::Vector{<:Number})::Vector{<:Number}
    output = input
    for layer in model.layers
        output = forward(layer, output)
    end
    return output
end

function backwardPropagation(model::Model, output::Vector{<:Number}, y::Vector{<:Number}, learning_rate::Float64 )
    grad = model.loss_func_prime.(y, output)
    for layer in reverse(model.layers)
        grad = backward(layer, grad, learning_rate)
    end
end

function predict(model::Model, input::Vector{<:Number})
    activations = forwardPropagation(model, input)
    return argmax(activations, dims=2)
end

function accuracy(y_pred::Vector{<:Number}, y_true::Vector{<:Number} )
    return sum(y_pred == y_true) / length(y_pred)
end