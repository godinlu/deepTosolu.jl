using Dash
using deepTosolu
using CSV
using DataFrames

data_train = CSV.File("./exemple/data/data_train.csv") |> DataFrame
# On extrait X_train et y_train
X_train = Matrix(select(data_train, [:x1_train, :x2_train]))
Y_train = data_train[:, :y_train]
data_test = CSV.File("./exemple/data/data_test.csv") |> DataFrame
# On extrait X_train et y_train
X_test = Matrix(select(data_test, [:x1_test, :x2_test]))
y_test = data_test[:, :y_test]

function training(X_train,Y_train,epochs,learning_rate,layers_txt,input_loss)
    layers = eval(Meta.parse(layers_txt))
    model = Model(layers)
    compile(model,input_loss)

    return fit(model, X_train, Y_train, epochs, learning_rate)

end

app = dash(external_stylesheets = ["https://codepen.io/chriddyp/pen/bWLwgP.css"])

    app.layout = html_div() do
        html_h1("DeepTosolu"),
        dcc_input(id = "layers_input", value = "[Dense(2,64),Sigmoid(),Dense(64,16),Sigmoid(),Dense(16,4),Sigmoid()]", type = "text",style = Dict("width" => "40%")),
        html_br(),
        dcc_input(id = "epoch_input", value = 100, type = "number"),
        dcc_input(id = "lr_input", value = 0.1, type = "number"),
        html_div(
            dcc_dropdown(
                id = "loss_input",
                options = [
                    Dict("label" => "Mean Squared Error", "value" => "mse"),
                    Dict("label" => "Binary Cross Entropy", "value" => "binaryCrossEntropy")
                ],
                value = "mse"  # Valeur par défaut sélectionnée
            )
        ),
        html_button("Valider", id = "submit-button"),
        html_div("Voici le graphique du loss"),
        dcc_graph(id = "loss", figure = ()),
        html_div("Voici le graphique de l'accuracy"),
        dcc_graph(id = "accur", figure = ())    
    end 

    callback!(app, Output("loss", "figure"),Output("accur","figure"), Input("submit-button", "n_clicks"),State("epoch_input", "value"),State("lr_input","value"),State("layers_input","value"),State("loss_input","value"),prevent_initial_call=true) do n_clicks,input_epoch,input_lr,input_layer,input_loss
        history = training(X_train,Y_train,input_epoch,input_lr,input_layer,input_loss)
        new_figure = (
            data = [
                (y = history["loss"], type = "scatter", mode = "lines", name = "Train loss"),
            ],
            layout = (title = "Courbe du loss",)
        )
        new_figure2 = (
                    data = [
                        (y = history["accuracy_train"], type = "scatter", mode = "lines", name = "accuracy train"),
                    ],
                    layout = (title = "Courbe de l'accuracy",)
                )
        return new_figure,new_figure2
    end

    run_server(app)
end