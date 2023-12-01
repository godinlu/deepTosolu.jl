using deepTosolu
using MLDatasets
using Plots

# Load training data (images, labels)
x_train, y_train = MNIST(split=:train)[:]

layers = [
    Convolutional((1,28,28), 3, 5),
    Sigmoid(),
    Flatten((5,26,26), (5*26*26, 1)),
    Dense(5*26*26, 100),
    Sigmoid(),
    Dense(100, 10),
    Sigmoid()
]

#on reshape les donné pour les adapté au réseau convolutif
x_train = reshape(x_train, (60000,1,28,28))
x_train = x_train/255



model = Model(layers)

compile(model, "binaryCrossEntropy")

history = fit(model, x_train, y_train, 5, 0.01)
plot(history["loss"])
plot(history["accuracy_train"])





print(x_train[:,:,:,1])

size(x_train)


