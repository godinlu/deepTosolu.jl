### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 20c65a90-7ed8-11ee-0a37-712092f75391
using Pkg; Pkg.add(url = "https://github.com/godinlu/deepTosolu.jl");Pkg.add("CSV");Pkg.add("DataFrames");Pkg.add("Plots")


# ╔═╡ 68bf286f-fffa-4ea5-8b9c-e662c7708051
begin
	using deepTosolu
	using CSV
	using DataFrames
	using Plots
	data_train = CSV.File("data_train.csv") |> DataFrame
	# On extrait X_train et y_train
	X_train = Matrix(select(data_train, [:x1_train, :x2_train]))
	y_train = data_train[:, :y_train]
	data_test = CSV.File("data_test.csv") |> DataFrame
	# On extrait X_train et y_train
	X_test = Matrix(select(data_test, [:x1_test, :x2_test]))
	y_test = data_test[:, :y_test]
	#create layers for the network
    layers = [
        Dense(2,64),
        Sigmoid(),
        Dense(64,64),
        Sigmoid(),
        Dense(64,4),
        Sigmoid()
    ]
    #create the model
    model = Model(layers)
    #compile the model with the mse loss function
    compile(model, "binaryCrossEntropy")
    #fit the model with the trainset
end

# ╔═╡ 368e8a5f-1b7a-4808-b8df-ca84b8f9db00
md"""
On commence par installer ce package de fou
"""

# ╔═╡ eff5c7c2-d4e3-46d8-8073-44ce10c1ae00
history = fit(model, X_train, y_train, 1000, 0.05)

# ╔═╡ ea74f5c8-6139-4632-a761-8cb8edac843b
plt_loss = plot(history["loss", ], title="binaryCrossEntropy loss")

# ╔═╡ f9de12fa-dbe3-4374-a5a0-6b7c80b0ad38
plt_accuracy = plot(history["accuracy_train"], color=:red, title="accuracy")

# ╔═╡ 38a4f6d7-2063-4937-a9b6-dcddf38dc92a
acc = evaluate(model, X_test, y_test)

# ╔═╡ ed634f23-32b9-4384-9b26-57dc4705a92d
println("accuracy : ",acc)

# ╔═╡ 3a167dba-082f-4f2c-b902-2be514584df8
md"""
!!! info "Attention !"
	On ne lance pas l'interface Dash ici car elle utilise Dash.jl,
	Cependant, on peut quand même afficher des graphiques
"""

# ╔═╡ c8ada748-384e-4b58-b90c-987a3b473f7e


# ╔═╡ 461743b0-d459-457a-8042-d9a0c1c6a56f


# ╔═╡ Cell order:
# ╟─368e8a5f-1b7a-4808-b8df-ca84b8f9db00
# ╠═20c65a90-7ed8-11ee-0a37-712092f75391
# ╠═68bf286f-fffa-4ea5-8b9c-e662c7708051
# ╠═eff5c7c2-d4e3-46d8-8073-44ce10c1ae00
# ╠═3a167dba-082f-4f2c-b902-2be514584df8
# ╠═ea74f5c8-6139-4632-a761-8cb8edac843b
# ╠═f9de12fa-dbe3-4374-a5a0-6b7c80b0ad38
# ╠═38a4f6d7-2063-4937-a9b6-dcddf38dc92a
# ╠═ed634f23-32b9-4384-9b26-57dc4705a92d
# ╠═c8ada748-384e-4b58-b90c-987a3b473f7e
# ╠═461743b0-d459-457a-8042-d9a0c1c6a56f
