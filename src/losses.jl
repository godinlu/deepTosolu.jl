function mse(y_true,y_pred)
    return mean((y_true - y_pred)^2)
end

function mse_prime(y_true,y_pred)
    return 2*(y_pred -y_true)/length(y_true)
end

function binary_cross_entropy(y_true, y_pred)
    return mean(-y_true * log(y_pred) - (1 -y_true)*log(1 - y_pred))
end

function binary_cross_entropy_prime(y_true, y_pred)
    return ((1 - y_true) / (1 - y_pred) - y_true / y_pred) / length(y_true)
end
