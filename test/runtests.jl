using deepTosolu
using Test

#deepTosolu.greet_your_package_name()
@testset "deepTosolu.jl" begin
dense = Dense(10,1)
output = forward(dense, rand(10))
println("output")

tanh = Tanh()
output = forward(tanh, rand(10))
println(output)


end
