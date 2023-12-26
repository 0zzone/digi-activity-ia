import pandas as pd 
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Préparer le jeu de données
train=pd.read_excel("./exam.xlsx") # On récupère le jeu de données
y = train.Class # On extrait la liste des réponses qu'on connaît déjà
train.drop(['Class'], axis=1, inplace=True) # On ne conserve que les autres colonnes pour traiter
X = train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=2) # Séparer notre jeu de données

# Calculs
DT= DecisionTreeClassifier() # Initialisation du modèle à utiliser
DT.fit(X_train,y_train) # Entraînement du modèle

# Prédiction pour un élève ayant 12 de moyenne et 30% de motivation
pred=DT.predict([[12,30]])
print(pred[0])
