
function Tanh()
    func = function(x::Number) tanh(x) end
    func_prime = function(x::Number) return (1- tanh(x)^2) end
    return Activation(func,func_prime)
end

function Sigmoid()
    func = function(x::Number) 1 / (1+exp(-x)) end
    func_prime = function (x::Number)
        s = func(x)
        return (s * (1-s))
    end
    sigmoid = Activation(func,func_prime)
    return sigmoid
end