mutable struct Model
    layers::Vector{Layer}
    loss_func::Function
    loss_func_prime::Function
    categories::Vector{Any}

end

Model(layers::Vector{Layer}) = Model(
    layers,
    mse,
    msePrime,
    Any[]
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

function ()
    
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
    grad = model.loss_func_prime(y, output)
    for layer in reverse(model.layers)
        grad = backward(layer, grad, learning_rate)
    end
end

function predict(model::Model, input::Vector{<:Number})
    activations = forwardPropagation(model, input)
    return model.categories[argmax(activations)] 
end

function evaluate(model::Model, X_test::Matrix{<:Number}, Y_test::Vector{<:Any})
    accuracy = 0 

    for i in 1:size(X_test, 1)
        accuracy += (predict(model,X_test[i,:]) == Y_test[i])
    end

    return accuracy / length(Y_test)
end

function accuracy(y_pred::Vector{<:Number}, y_true::Vector{<:Number} )
    return sum(y_pred == y_true) / length(y_pred)
end

function oneHotEncode(model::Model, vector::Vector{<:Any})::Matrix{Int}
    categArray = CategoricalArray(vector)
    model.categories = levels(categArray)
    one_hot_matrix = [x==l for x in categArray, l in model.categories ]
    return one_hot_matrix

end

function fit(
    model::Model,
    X_train::Matrix{<:Number},
    Y_train::Vector{<:Any},
    epochs::Int,
    learning_rate::Float64 = 0.1
)

    loss = zeros(epochs)
    acc_train = zeros(epochs)
    Y_train_oh = oneHotEncode(model, Y_train)
    pbar = ProgressBar(total = epochs * size(X_train,1))

    for e in 1:epochs
        error = 0
        tmp_acc = 0
        for i in 1:size(X_train,1)
            x = X_train[i,:]
            y = Y_train_oh[i,:]

            output = forwardPropagation(model, x)

            error = error + model.loss_func(y, output)

            backwardPropagation(model, output, y, learning_rate)

            tmp_acc = tmp_acc + (predict(model, x) == Y_train[i])
            update(pbar)
        end

        loss[e] = error / length(X_train)
        acc_train[e] = tmp_acc / size(X_train, 1)
    end

    history = Dict(
        "loss" => loss,
        "accuracy_train" => acc_train
    )
    return history
end