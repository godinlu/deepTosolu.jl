
function Tanh()
    func = function(x::Float64) tanh(x) end
    func_prime = function(x::Float64) 1- tanh(x)^2 end
    return Activation(func,func_prime)
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