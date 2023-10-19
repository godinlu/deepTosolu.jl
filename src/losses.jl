function mse(y_true::Number,y_pred::Number)
    return mean((y_true - y_pred)^2)
end

function msePrime(y_true::Number,y_pred::Number)
    return 2*(y_pred -y_true)/length(y_true)
end

function binaryCrossEntropy(y_true::Number, y_pred::Number)
    return mean(-y_true * log(y_pred) - (1 -y_true)*log(1 - y_pred))
end

function binaryCrossEntropyPrime(y_true::Number, y_pred::Number)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / length(y_true)
end
