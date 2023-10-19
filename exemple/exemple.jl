using  deepTosolu


dense = Dense(10,1)
output = forward(dense, rand(10))


tanh = Tanh()
output = forward(tanh, rand(10))

sig = Sigmoid()
output_sig = forward(sig, rand(10))


X = [0 0; 0 1; 1 0; 1 1]  # Les quatre combinaisons possibles d'entrées XOR
Y = [0; 1; 1; 0]          # Les résultats attendus pour chaque combinaison XOR


layers = [
    Dense(2,3),
    Tanh(),
    Dense(3,1),
    Tanh()
]
model = Model(layers)
compile(model,"mse")
output = forwardPropagation(model, X[1,:])
backwardPropagation(model, output, Y[1,:], 0.1)