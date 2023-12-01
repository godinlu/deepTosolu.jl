using deepTosolu
using Random

#On commence par créer une image de 28 * 28 aléatoire
img = randn((1,28,28))

# ensuite on créer une couche de convolution qui a comme paramètre la dimension
# de l'image soit (1,28,28) ensuite la taille de noyau de convolution en général on choisit 3,
# puis la profondeur qui correspond au nombre d'image en sortie dans notre cas 5
convolution1 = Convolutional((1, 28, 28), 3, 5)

# Ensuite on peut tester la méthode forward de la convolution pour voir, 
# sachant que lorsque l'on applique une convolution à une image on est censé 
# observé une réduction de la taille de l'image lié à la taille du noyau 
output1 = forward(convolution1, img)

#on observe la shape suivante : (5, 26, 26) ce qui correspond bien à ce que nous attendons
print(size(output1))

#si l'on veut ajouter une deuxième couche de convolution il faudra donc la shape suivante 
convolution2 = Convolutional((5, 26, 26), 3, 8)

output2 = forward(convolution2, output1)
#on a bien la shape suivante (8, 24, 24)
print(size(output2))

#une fois les convolutions faite il faut ajouter une couche de flatten pour applatir les images et les passé 
# dans un réseau de neuronne dense
flatten = Flatten((8,24,24), (8*24*24,1))

output3 = forward(flatten, output2)
print(size(output3))

#maintenant observont la bacward propagation pour voir si l'on retrouve bien la bonne size de l'image
input1 = backward(flatten, output3, 0.1)
input2 = backward(convolution2, input1, 0.1)
input3 = backward(convolution1, input2, 0.1)

#on retrouve bien la shape d'origine soit (1, 28, 28) donc
#théoriquement la backward propagation marche aussi
print(size(input3))

# le seul problème est le suivant: normalement après une couche de convolution il faut il fonction d'activation
# mais pour l'instant les fonctions d'activations prennent en parmètre des vecteur et non des matrice
# si ce problème est résolue potentiellement les convolutions peuvent marcher.






