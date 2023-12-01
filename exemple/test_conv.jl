using deepTosolu
using Random


convolution1 = Convolutional((1, 28, 28), 3, 5)
convolution2 = Convolutional((5, 26, 26), 3, 8)

input = randn((1,28,28))

f = forward(convolution1, input)

f = forward(convolution2, f)

println(size(f))
#backward(convolution, f, 0.1)



#test de la couche reshape
# reshape = Reshape((5,28,28) , (5*28*28, 1))
# f = forward(reshape, f)

M = rand(5,5,5)


layers = [
        Dense(4,5),
        Sigmoid(),
        Dense(5,2),
        Sigmoid()
    ]

