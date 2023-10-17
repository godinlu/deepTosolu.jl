
function Tanh()
    func = function(x) tanh(x) end
    func_prime = function(x) 1- tanh(x)^2 end
    tanh = Activation(func,func_prime)
    return tanh
end

function Sigmoid()
    func = function(x) 1 / (1+exp(-x)) end
    func_prime = function (x)
        s = func(x)
        return (s * (1-s))
    end
    sigmoid = Activation(func,func_prime)
    return sigmoid
end