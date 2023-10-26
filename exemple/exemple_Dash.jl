using Dash
using  deepTosolu
using Plots
using Random

# Générer les données d'entraînement
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

#scatter(X_train[:,1],X_train[:,2], mc=Y_train)



layers = [
    Dense(2,64),
    Sigmoid(),
    Dense(64,3),
    Sigmoid(),
    Dense(3,3),
    Sigmoid()
]
model = Model(layers)
compile(model,"mse")


history = fit(model, X_train, Y_train, 100, 0.1)

for i in 1:size(X_test, 1)
    x = X_test[i,:]
    #println(forwardPropagation(model, x))
    println(predict(model,x))
    #println(forwardPropagation(model, ))
end
plot(history["loss"])
plot(history["accuracy_train"])



app = dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])

app.layout = html_div() do
    html_h1("Hello Dash"),
    html_div("Dash.jl: Julia interface for Dash"),
    dcc_graph(id = "example-graph",
              figure = (
                  data = [
                      (x = [1, 2, 3], y = [4, 1, 2], type = "bar", name = "SF"),
                      (x = [1, 2, 3], y = [2, 4, 5], type = "bar", name = "Montréal"),
                  ],
                  layout = (title = "Dash Data Visualization",)
              ))
end

run_server(app)