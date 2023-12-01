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
    #On créé le modèle
    model = Model(layers)
    #On le compile en choisissant une fonction de loss
    compile(model, "binaryCrossEntropy")
end

# ╔═╡ 368e8a5f-1b7a-4808-b8df-ca84b8f9db00
md"""
On commence par installer le package en utilisant le git
"""

# ╔═╡ e6bea3f3-df8b-45a2-ab0c-8f0eca56ade9
md"""
Maintenant que le modèle existe, on peut l'entraîner. On en profite pour mesurer le temps d'éxécution pour le comparer avec la version du programme en R
"""

# ╔═╡ eff5c7c2-d4e3-46d8-8073-44ce10c1ae00
begin 
	elapsed_time = @elapsed begin
    	history = fit(model, X_train, y_train, 1000, 0.1);
	end
	println("Temps écoulé : $elapsed_time secondes")
end

# ╔═╡ 3a167dba-082f-4f2c-b902-2be514584df8
md"""
!!! info "Attention !"
	On ne lance pas l'interface Dash ici car elle utilise Dash.jl,
	Cependant, on peut quand même afficher des graphiques
"""

# ╔═╡ ea74f5c8-6139-4632-a761-8cb8edac843b
plt_loss = plot(history["loss", ], title="binaryCrossEntropy loss")

# ╔═╡ 5e4f0040-68bf-45a0-bfed-81755fc14a59
md"""
On peut observer les résultats dans les différents graphiques
"""

# ╔═╡ f9de12fa-dbe3-4374-a5a0-6b7c80b0ad38
plt_accuracy = plot(history["accuracy_train"], color=:red, title="accuracy")

# ╔═╡ Cell order:
# ╟─368e8a5f-1b7a-4808-b8df-ca84b8f9db00
# ╠═20c65a90-7ed8-11ee-0a37-712092f75391
# ╠═68bf286f-fffa-4ea5-8b9c-e662c7708051
# ╟─e6bea3f3-df8b-45a2-ab0c-8f0eca56ade9
# ╠═eff5c7c2-d4e3-46d8-8073-44ce10c1ae00
# ╟─3a167dba-082f-4f2c-b902-2be514584df8
# ╟─ea74f5c8-6139-4632-a761-8cb8edac843b
# ╟─5e4f0040-68bf-45a0-bfed-81755fc14a59
# ╟─f9de12fa-dbe3-4374-a5a0-6b7c80b0ad38
