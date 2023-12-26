from main import main_function
from skimage.io import imread
from skimage.transform import resize
import os

# On récupère les variables suivantes

# sgd_clf -> notre modèle d'IA
# grayify -> mettre en gray l'image pour la traiter
# hogify -> traiter les "gradient" de l'image
# scalify -> permet de standardiser les attributs des images
sgd_clf, grayify, hogify, scalify = main_function()

images = [] # Les images qu'on voudra traiter seront mises dans ce tableau

image_a_classifier = imread(os.path.join(os.getcwd(), 'test.jpg')) # Récupérer l'image
image_a_classifier = image_a_classifier[:, :, :3] # On doit garder que les filtres utiles pour notre image
image_a_classifier = resize(image_a_classifier, (80, 80)) # On doit modifier la taille de notre image

images.append(image_a_classifier) # Ajouter les images qu'on veut traiter (ici une seule !)


image_a_classifier_gray = grayify.transform(images) # Gray
image_a_classifier_hog = hogify.transform(image_a_classifier_gray) # Hog
image_a_classifier_prepared = scalify.transform(image_a_classifier_hog) # scale

prediction = sgd_clf.predict(image_a_classifier_prepared) # On prédit !

print(prediction) # On affiche la prédiction