### A Pluto.jl notebook ###
# v0.19.32

using Markdown
using InteractiveUtils

# ╔═╡ 20c65a90-7ed8-11ee-0a37-712092f75391
using Pkg; Pkg.add(url = "https://github.com/godinlu/deepTosolu.jl")

# ╔═╡ 68bf286f-fffa-4ea5-8b9c-e662c7708051
begin
	using deepTosolu
	using Random
	Random.seed!(42)  # Pour la reproductibilité
	
	x1_train = vcat(randn(50) * 2 .+ 5, randn(50) * 2 .- 5, randn(50) * 2)
	x2_train = vcat(randn(50) * 2 .+ 5, randn(50) * 5 .- 5, randn(50) * 2)
	X_train = hcat(x1_train, x2_train)
	
	# Générer les données de test
	x1_test = vcat(randn(50) * 2 .+ 5, randn(50) * 2 .- 5, randn(50) * 2)
	x2_test = vcat(randn(50) * 2 .+ 5, randn(50) * 5 .- 5, randn(50) * 2)
	X_test = hcat(x1_test, x2_test)
	
	# Créer les étiquettes (Y_train et Y_test)
	Y_train = vcat(fill("chat",50),  fill("chien",50), fill("camion",50))
	Y_test = vcat(fill("chat",50),  fill("chien",50), fill("camion",50))
	
	#interface(X_train,Y_train)
end

# ╔═╡ 368e8a5f-1b7a-4808-b8df-ca84b8f9db00
md"""
On commence par installer ce package de fou
"""

# ╔═╡ 3a167dba-082f-4f2c-b902-2be514584df8
md"""
!!! info "Attention !"
	On ne lance pas l'interface ici car elle utilise Dash.jl
	Cependant, on peut quand même afficher des graphiques
"""

# ╔═╡ c8ada748-384e-4b58-b90c-987a3b473f7e


# ╔═╡ 461743b0-d459-457a-8042-d9a0c1c6a56f


# ╔═╡ Cell order:
# ╟─368e8a5f-1b7a-4808-b8df-ca84b8f9db00
# ╠═20c65a90-7ed8-11ee-0a37-712092f75391
# ╠═68bf286f-fffa-4ea5-8b9c-e662c7708051
# ╠═3a167dba-082f-4f2c-b902-2be514584df8
# ╠═c8ada748-384e-4b58-b90c-987a3b473f7e
# ╠═461743b0-d459-457a-8042-d9a0c1c6a56f
