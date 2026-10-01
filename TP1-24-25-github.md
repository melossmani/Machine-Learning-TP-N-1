# 🤖 Machine Learning — TP N°1

> **4ème année GI-IADS · 2024/2025**  
> **Enseignant : Mustapha El Ossmani**

## 📚 Sommaire

- [Objectifs du TP](#-objectifs-du-tp)
- [Partie 1 — Régression linéaire simple avec NumPy](#-partie-1--régression-linéaire-simple-avec-numpy)
  - [1.1 Dataset](#11-dataset)
  - [1.2 Visualisation des données](#12-visualisation-des-données)
  - [1.3 Matrice X](#13-matrice-x)
  - [1.4 Vecteur de paramètres](#14-vecteur-de-paramètres)
  - [1.5 Modèle linéaire](#15-modèle-linéaire)
  - [1.6 Fonction coût](#16-fonction-coût)
  - [1.7 Gradient et descente de gradient](#17-gradient-et-descente-de-gradient)
  - [1.8 Phase d'entraînement](#18-phase-dentraînement)
  - [1.9 Courbes d'apprentissage](#19-courbes-dapprentissage)
  - [1.10 Évaluation finale](#110-évaluation-finale)
- [Partie 2 — Régression multiple et polynomiale avec NumPy](#-partie-2--régression-multiple-et-polynomiale-avec-numpy)
  - [2.1 Régression polynomiale](#21-régression-polynomiale)
  - [2.2 Régression à plusieurs variables](#22-régression-à-plusieurs-variables)
- [Partie 3 — Régression linéaire avec scikit-learn](#-partie-3--régression-linéaire-avec-la-librairie-scikit-learn)

---

# Machine Learning — TP N°1 (2024/2025)

**4ème année GI-IADS**  
**Enseignant : Mustapha El Ossmani**

## 🎯 Objectifs du TP

Charger les données pour la régression

Simulé des données par `make_regression` du module `sklearn.datasets`

Mettre en œuvre le modèle de régression linéaire par

Equations normales

Descente de gradient (effet du learning rate)

`LinearRegression` du module `sklearn.linear_model`

Faire des figures dans le cas 1D et 2D

Calculer le coefficient de performance  `r2_score` manuellement avec numpy

Séparer les données en trainset et testset par `train_test_split` du module `sklearn.model_selection`

Evaluer la qualité du modèle de régression par `mean_squared_error` et `r2_score` du module `sklearn.metrics`

## Partie 1 : Régression Linéaire Simple Numpy

```python
import numpy as np
from `sklearn.datasets` import `make_regression`
```

```python
import `matplotlib`.pyplot as plt
```

### 1.1 Dataset

Génération de données aléatoires avec une tendance linéaire avec `make_regression`: on a un dataset (x,y) qui contient 100 exemples, et une seule variable x.

> **Note :** chaque fois que la cellule est executée, des données différentes sont générées. Utiliser np.random.seed(0) pour reproduire le même Dataset à chaque fois.

Récupérer les features x et le target y.

np.random.seed(0) # pour toujours reproduire le meme dataset

x, y = `make_regression`(n_samples=100, n_features=1, noise=10)

Les commentaires sous python sont précédés par le symbole #.

> **Important :** vérifier les dimensions de x et y. On remarque que y n'a pas les dimensions (100, 1). On corrige le problème avec `np.reshape`

```python
print(x.shape)
print(y.shape)
```

```python
# redimensionner y
y = y.reshape(y.shape[0], 1)
```

```python
print(y.shape)
```

### 1.2 Visualisation des données

```python
# afficher les résultats. x en abscisse et y en ordonnée
plt.scatter(x, y) 
```

```python
plt.xlabel ("x")
plt.ylabel (" y")
```

```python
plt.title(" Le Titre")
plt.show()
```

### 1.3 Matrice X

```python
X = `np.hstack`((`np.ones`(x.shape),x))
print(X.shape)
```

### 1.4 Vecteur de paramètres

np.random.seed(0) # pour produire toujours le même vecteur a aléatoire

```python
a = `np.random.randn`(2, 1)
```

a

### 1.5 Modèle linéaire :  On implémente un modèle 𝐹=𝑋.a, puis on teste le modèle pour voir s'il n'y a pas de bug (bonne pratique oblige). En plus, cela permet de voir à quoi ressemble le modèle initial, défini par la valeur de a

```python
def model(X, a):
```

return X.dot(a)

```python
plt.scatter(x, y)
plt.plot(x, model(X, a), c='r')
```

### 1.6 Fonction coût — erreur quadratique moyenne

On mesure les erreurs du modele sur le Dataset X, y en implémenterl'erreur quadratique moyenne, Mean Squared Error (MSE) en anglais.

Ensuite, on teste notre fonction, pour voir s'il n'y a pas de bug.

```python
def cost_function(X, y, a):
    m = len(y)
```

return 1/(2*m) * np.sum((model(X, a) - y)**2)

```python
cost_function(X, y, a)
```

### 1.7 Gradient et descente de gradient

On implémente la formule du gradient pour la MSE :

Ensuite on utilise cette fonction dans la descente de gradient:

```python
def grad(X, y, a):
    m = len(y)
```

return 1/m * X.T.dot(model(X, a) - y)

```python
def gradient_descent(X, y, a, learning_rate, n_iterations):
     # création d'un tableau de stockage pour enregistrer l'évolution du Cout du modele
```

```python
    cost_history = np.zeros(n_iterations) 
```

for i in range(0, n_iterations):

```python
	   # mise a jour du parametre theta (formule du gradient descent)
        a = a - learning_rate * grad(X, y, a) 
```

cost_history[i] = cost_function(X, y, a)

```python
           # on enregistre la valeur du Cout au tour i dans cost_history[i]
```

return a, cost_history

### 1.8 Phase d'entraînement

On définit un nombre d'itérations, ainsi qu'un pas d'apprentissage 𝛼.

Une fois le modele entrainé, on observe les résultats par rapport à notre Dataset

```python
n_iterations = 1000
learning_rate = 0.01
```

theta_final, cost_history = gradient_descent(X, y, a, learning_rate, n_iterations)

```python
# voici les parametres du modele une fois que la machine a été entrainée
```

theta_final

```python
# création d'un vecteur prédictions qui contient les prédictions de notre modele final
predictions = model(X, theta_final)
```

```python
# Affiche les résultats de prédictions (en rouge) par rapport a notre Dataset (en bleu)
plt.scatter(x, y)
```

```python
plt.plot(x, predictions, c='r')
```

### 1.9 Courbes d'apprentissage

Pour vérifier si notre algorithme de Descente de gradient a bien fonctionné, on observe l'évolution de la fonction cout a travers les itérations. On est sensé obtenir une courbe qui diminue a chaque itération jusqu'a stagner a un niveau minimal (proche de zéro). Si la courbe ne suit pas ce motif, alors le pas learning_rate est peut-etre trop élevé, il faut prendre un pas plus faible.

Pour cela, utiliser la variable cost_history pour faire ce plot.

```python
plt.plot()
```

### 1.10 Évaluation finale

Pour évaluer la réelle performance de notre modele avec une métrique populaire, on peut utiliser le coefficient de détermination, aussi connu sous le nom R2. Il nous vient de la méthode des moindres carrés. Plus le résultat est proche de 1, meilleur est votre modèle.

Définire une fonction qui retourne  R2 .

```python
def coef_determination(y, pred):
    u = 
```

```python
    v = 
```

return 1 - u/v

```python
coef_determination(y, predictions)# afficher le coefficient de performan
```

## Partie 2 : Régression Linéaire Multiple et Polynomiale Numpy

### 2.1 Régression polynomiale — une variable x₁

#### 2.1.1 Dataset

Pour développer un modèle polynomial à partir des équations de la régression linéaire, il suffit d'ajouter des degrés de polynome dans les colonnes de la matrice X ainsi qu'un nombre égal de lignes dans le vecteur a.

Ici, nous allons développer un polynôme de degré 2:  𝑓(x)=a2x2+a1x+a0. Pour celà, il faut développer les matrices suivantes:

> **note  :** le vecteur 𝑦 reste le meme que pour la régression linéaire

np.random.seed(0) # permet de reproduire l'aléatoire

```python
# creation d'un dataset (x, y) linéaire
```

x, y = `make_regression`(n_samples=100, n_features=1, noise = 10)

```python
# modifie les valeurs de y pour rendre le dataset non-linéaire
y = y + abs(y/2) 
```

```python
plt.scatter(x, y) # afficher les résultats. x en abscisse et y en ordonnée
```

Question : Comment peut-on utiliser la première partie pour faire la régression polynomiale? ( suivre les mêmes démarches que la partie 1)

### 2.2 Régression à plusieurs variables

C'est lorsqu'on intègre plusieurs variables  x1, x2, x3, 𝑒𝑡𝑐.

à notre modèle que les choses commencent à devenir vraiment intéressantes. C'est peut-être aussi à ce moment que les gens commencent parfois à parler d'intelligence artificielle, car il est difficile pour un être humain de se représenter dans sa tête un modèle à plusieurs dimensions (nous n'évoluons que dans un espace 3D). On se dit alors que la machine, quant à elle, arrive à se représenter ces espaces, car elle y trouve le meilleur modèle (avec la descente de gradient) et les gens disent donc qu'elle est intelligente, alors que ce ne sont que des mathématiques.

#### 2.2.1 Dataset

Maintenant, nous allons créer un modèle à 2 variables x1, x2. Pour cela, il suffit d'injecter les différentes variables x1, x2 (les features en anglais) dans la matrice X, et de créer le vecteur a qui s'accorde avec:

> **note  :** le vecteur 𝑦 reste le meme que pour la régression linéaire.

np.random.seed(0) # permet de reproduire l'aléatoire

```python
# creation d'un dataset (x, y) linéaire
```

x, y = `make_regression`(n_samples=100, n_features=2, noise = 10)

```python
# afficher les résultats. x_1 en abscisse et y en ordonnée
plt.scatter(x[:,0], y) 
```

Ce Dataset ne contenant que 2 variables x1 𝑒𝑡 x2,  il est possible de le visualiser dans un espace 3D. Comme vous pouvez le voir, ce modèle peut être représenté par une surface. Au passage, cette surface est plane car `make_regression` nous retourne des données linéaires. Si on veut créer une surface non plane, il suffit de modifier la valeur de y comme nous l'avons fait au début.

```python
from mpl_toolkits.mplot3d import Axes3D
#%`matplotlib` notebook #activez cette ligne pour manipuler le graph 3D
```

```python
ax = fig.add_subplot(111, projection='3d')
ax.scatter(x[:,0], x[:,1], y) # affiche en 3D la variable x_1, x_2, et la target y
```

```python
# affiche les noms des axes
ax.set_xlabel('x_1')
```

```python
ax.set_ylabel('x_2')
ax.set_zlabel('y')
```

Question : Généraliser la partie 1 pour deux features.

---

## 🧰 Partie 3 — Régression linéaire avec la librairie scikit-learn

La méthode de régression linéaire est implémentée dans la librairie `sklearn.linear_model` sous le nom `LinearRegression`. Utiliser cette fonction pour programmer la régression linéaire de vos données, puis évaluer la qualité de votre modèle.

```python
from `sklearn.linear_model` import `LinearRegression`
model = `LinearRegression`()
```

```python
model.fit(x, y)  # apprentissage
y_pred = model.predict(x) # prediction
```

La fonction `PolynomialFeatures` du module `sklearn.preprocessing` génère une nouvelle matrice de features composée de toutes les combinaisons polynômiales des features de degré inférieur ou égale au degré indiqué. Pour degree=2 avec deux features x1 et x2, cette fonction renvoie [1, x1, x2, x1x2, x12, x22].

```python
from `sklearn.preprocessing` import `PolynomialFeatures`
polynomial_features= `PolynomialFeatures`(degree=1) # polynomial degree
```

```python
X = polynomial_features.fit_transform(x)
```

Nous utilisons aussi la fonction `train_test_split` de la librairie `sklearn.model_selection` pour séparer les données en deux parties : une partie pour le training et l’autre pour le test.

```python
from `sklearn.model_selection` import `train_test_split`
```

x_train,x_test,y_train,y_test = `train_test_split`(x, y, random_state=4)

On considère les données non linéaires suivantes :

np.random.seed(0)

```python
x = np.sort(7 * np.random.rand(80, 1), axis=0)
y = np.sin(x).ravel() + 0.1*np.random.normal(0,1,len(x))
```

Le but est d’appliquer ce que nous avons vu dans les deux parties précédentes.

On suit le plan suivant :

Visualiser les données

Séparer les données en deux parties (training set and test set)

Faire l’apprentissage du modèle `LinearRegression` sur les données du training set

Evaluer le modèle obtenu sur les données du test set

Faire augmenter le degré du polynôme, puis refaire l’apprentissage et l’évaluation

Comparer les résultats des modèles obtenus

Visualiser les données de training set et testset de deux couleurs différentes

Tracer les courbes de régression correspondantes à chaque modèle.

---

## 💡 Remarque

Ce document reprend le contenu du TP et est structuré en Markdown pour une publication et une lecture plus confortables sur GitHub.
