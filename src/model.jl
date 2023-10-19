mutable struct Model
    layers::Vector{Layer}
    loss_func::Function
    loss_func_prime::Function

end

Model(layers::Vector{Layer}) = Model(
    layers,
    nothing,
    nothing
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
