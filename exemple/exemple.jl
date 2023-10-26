using  deepTosolu
using Plots
using Random


function createDataSet()
    # Générer les données d'entraînement
    Random.seed!(40)  # Pour la reproductibilité

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

    return Dict(
        "X_train" => X_train,
        "Y_train" => Y_train,
        "X_test" => X_test,
        "Y_test" => Y_test
    )
end



function main()
    #get a dict contain the dataSet
    dataset = createData()

    #create layers for the network
    layers = [
        Dense(2,64),
        Sigmoid(),
        Dense(64,3),
        Sigmoid(),
        Dense(3,3),
        Sigmoid()
    ]
    #create the model
    model = Model(layers)
    #compile the model with the mse loss function
    compile(model, "binaryCrossEntropy")
    #fit the model with the trainset
    history = fit(model, dataset["X_train"], dataset["Y_train"], 300, 0.1)
    
    plt_loss = plot(history["loss"])
    plt_accuracy = plot(history["accuracy_train"])

    display(plot(plt_loss, plt_accuracy, layout=2))
    acc = evaluate(model, dataset["X_test"], dataset["Y_test"])
    println("accuracy : ",acc)

end

main()



