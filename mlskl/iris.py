# Introduction à SciKitLearn 
# IRIS : 150 observations de 3 types de fleurs d'iris, avec pour chacune d'elles quatre caractéristiques (longueur et largeur des pétales, longueur et largeur des sépales) et le nom de la variété d'iris. Le cas d'usage habituel est de créer des modèles permettant de prédire la variété en fonction des caractéristiques observées. 
# https://khayyam.developpez.com/articles/machine-learning/scikit-learn/ 

print("----- Import du dataset IRIS -----")
# import some data to play with
from sklearn import datasets
iris = datasets.load_iris()
X = iris.data[:, :2]  # we only take the first two features.
Y = iris.target



print("----- Découpage -----")
from sklearn.model_selection import train_test_split
# L'ensemble de test aura 20 % des éléments de départ. 
# L'ensemble d'entrainement contiendra les 80 % restant.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)

print("----- Classification : k plus proches voisins -----")
from sklearn.neighbors import KNeighborsClassifier

model = KNeighborsClassifier(n_neighbors = 5)
model.fit(X_train, Y_train)   
y_pred = model.predict(X_test)

print("----- Regression -----")
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

degre = 2

# création d'un ensemble X,Y clairement associé à une parabole
X = np.linspace(1, 10, 100)  # création de 100 points en abscisse
Y = X**2 + 10*np.random.rand((100))   # création de 100 points en ordonnée

X = X.reshape((100, 1))
Y = Y.reshape((100, 1))
    
model = make_pipeline(PolynomialFeatures(degre), Ridge())
model.fit(X, Y)
Y_pred = model.predict(X)

# graphique du résultat
plt.scatter(X, Y, c="r")
plt.plot(X, Y_pred, c="b", lw=5)
plt.show()


print("----- PCA : Principal Component Analysis -----")
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_r = pca.fit(X).transform(X)


