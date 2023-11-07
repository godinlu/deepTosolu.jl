using deepTosolu
using MLDatasets
using Plots

# Load training data (images, labels)
x_train, y_train = MNIST(split=:train)[:]

# applatie les images (flatten)
x_train = reshape(x_train, (28*28,60000))

x_train = Matrix(x_train')

layers = [
        Dense(28*28,16),
        Sigmoid(),
        Dense(16,16),
        Sigmoid(),
        Dense(16,16),
        Sigmoid(),
        Dense(16,10),
        Sigmoid()
    ]
model = Model(layers)

compile(model, "binaryCrossEntropy")

history = fit(model, x_train, y_train, 5, 0.01)
plot(history["loss"])
plot(history["accuracy_train"])

