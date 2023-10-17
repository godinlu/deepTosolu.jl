abstract type Layer end

forward(layer::Layer, args...) = error("layer is an abstract class")

backward(layer::Layer, args...) = error("layer is an abstract class")

