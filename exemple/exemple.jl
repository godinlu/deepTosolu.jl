using  deepTosolu


dense = Dense(10,1)
output = forward(dense, rand(10))


tanh = Tanh()
output = forward(tanh, rand(10))
println(output)