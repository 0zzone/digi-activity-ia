import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split 
from math import *

# Immporter notre dataset (l'ensemble de nos données)
dataset = pd.read_excel('notes.xls')

heures = dataset.iloc[:,:-1] # On récupère les heures
notes = dataset.iloc[:, 1] # On récupère les notes

# Séparer notre jeu de données en deux (Test & Entraînement)
x_train, x_test, y_train, y_test = train_test_split(heures, notes, test_size=0.3, random_state=0)

plt.scatter(heures, notes, color='red')
plt.title('Notes obtenues')
plt.xlabel('Heures')
plt.ylabel('Notes')

Regressor = LinearRegression() # On créé l'objet de régression
Regressor.fit(x_train, y_train) # On entraîne notre jeu de données
y_pred = Regressor.predict(x_test) # On prédit les valeurs de sortie

x = np.linspace(1,35)
y = [Regressor.coef_ * item + Regressor._residues for item in x]
plt.plot(x, y, color='orange')
plt.show()