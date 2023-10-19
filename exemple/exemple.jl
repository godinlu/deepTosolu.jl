using  deepTosolu


dense = Dense(10,1)
output = forward(dense, rand(10))


tanh = Tanh()
output = forward(tanh, rand(10))
println(output)

sig = Sigmoid()
output_sig = forward(sig, rand(10))
println(output_sig)