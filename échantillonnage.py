import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Set a random seed for reproducibility
population = np.random.normal(loc=170, scale=10, size=10000)  # Create a normal distribution with mean 170 and standard deviation 10

echantillon = np.random.choice(population, size=100, replace=False)

plt.figure(figsize=(12, 6))  # Create a figure with a size of 12 inches by 6 inches

# Plot the population distribution
plt.subplot(1, 2, 1)  # Create a subplot in the first row, first column
plt.hist(population, bins=50, density=True, alpha=0.7, color='blue')  # Plot a histogram of the population
plt.title('Distribution de la Population')  # Set the title of the plot
plt.xlabel('Taille (cm)')  # Set the x-axis label
plt.ylabel('Densité')  # Set the y-axis label

# Plot the sample distribution
plt.subplot(1, 2, 2)  # Create a subplot in the first row, second column
plt.hist(echantillon, bins=20, density=True, alpha=0.7, color='green')  # Plot a histogram of the sample
plt.title('Distribution de l\'Échantillon')  # Set the title of the plot
plt.xlabel('Taille (cm)')  # Set the x-axis label
plt.ylabel('Densité')  # Set the y-axis label

plt.tight_layout()  # Adjust the layout of the subplots
plt.show()  # Display the plot

##################################################################################################

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Set a random seed for reproducibility
population = np.random.normal(loc=170, scale=10, size=10000)  # Create a normal distribution with mean 170 and standard deviation 10

echantillon = np.random.choice(population, size=100, replace=False)

plt.figure(figsize=(12, 6))  # Create a figure with a size of 12 inches by 6 inches

# Plot the population distribution
plt.subplot(1, 2, 1)  # Create a subplot in the first row, first column
plt.hist(population, bins=50, density=True, alpha=0.7, color='blue')  # Plot a histogram of the population
plt.title('Distribution de la Population')  # Set the title of the plot
plt.xlabel('Taille (cm)')  # Set the x-axis label
plt.ylabel('Densité')  # Set the y-axis label

# Plot the sample distribution
plt.subplot(1, 2, 2)  # Create a subplot in the first row, second column
plt.hist(echantillon, bins=20, density=True, alpha=0.7, color='green')  # Plot a histogram of the sample
plt.title('Distribution de l\'Échantillon')  # Set the title of the plot
plt.xlabel('Taille (cm)')  # Set the x-axis label
plt.ylabel('Densité')  # Set the y-axis label

plt.tight_layout()  # Adjust the layout of the subplots
plt.show()  # Display the plot


##################################################################################################


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Set a random seed for reproducibility
population = np.random.uniform(low=50, high=100, size=100000)  # Population uniforme entre 50 et 100

n_simulations = 1000  # Nombre de simulations
taille_echantillon = 30  # Taille de chaque échantillon

z_values = np.zeros(n_simulations)  # Tableau pour stocker les valeurs normalisées Z_n

for i in range(n_simulations):
    # Prélèvement d'un échantillon
    echantillon = np.random.choice(population, size=taille_echantillon, replace=False)

    S_n = np.sum(echantillon)  # Somme des valeurs dans l'échantillon

    Z_n = (S_n - taille_echantillon * mu) / (sigma * np.sqrt(taille_echantillon))  # Normalisation

    z_values[i] = Z_n  # Stocker la valeur de Z_n

plt.figure(figsize=(12, 6))  # Create a figure with a size of 12 inches by 6 inches

# Distribution des valeurs normalisées Z_n
plt.hist(z_values, bins=30, density=True, alpha=0.7, color='green', label='Distribution de Z_n')

# Superposition avec la courbe de densité de N(0,1)
x = np.linspace(-4, 4, 100)
phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)
plt.plot(x, phi, color='red', linestyle='--', label='N(0,1)')

plt.title('Théorème Central Limite : Distribution de Z_n')
plt.xlabel('Z_n')
plt.ylabel('Densité')
plt.legend()
plt.show()

z_mean = np.mean(z_values)
z_std = np.std(z_values)

print(f"Moyenne de Z_n (attendue = 0) = {z_mean:.2f}")
print(f"Écart-type de Z_n (attendu = 1) = {z_std:.2f}")


##################################################################################################


import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)  # Assure la reproductibilité
population = np.random.uniform(low=50, high=100, size=100000)  # Population uniforme entre 50 et 100

mu = np.mean(population)  # Moyenne réelle de la population
sigma = np.std(population)  # Écart-type réel de la population

n_simulations = 1000  # Nombre de simulations
taille_echantillon = 30  # Taille de chaque échantillon

moyennes = np.zeros(n_simulations)  # Tableau pour stocker les moyennes des échantillons
ecarts_types = np.zeros(n_simulations)  # Écarts-types des échantillons

for i in range(n_simulations):
    echantillon = np.random.choice(population, size=taille_echantillon, replace=False)

    moyennes[i] = np.mean(echantillon)  # Calcul de la moyenne de chaque échantillon
    ecarts_types[i] = np.std(echantillon, ddof=1)  # Écart-type de l'échantillon (ddof=1 pour l'échantillon)

moyenne_estimee = np.mean(moyennes)
ecart_type_estime = np.mean(ecarts_types)

print(f"Moyenne réelle de la population : {mu:.2f}")
print(f"Moyenne estimée (moyenne des moyennes des échantillons) : {moyenne_estimee:.2f}\n")

print(f"Écart-type réel de la population : {sigma:.2f}")
print(f"Écart-type estimé (moyenne des écarts-types des échantillons) : {ecart_type_estime:.2f}\n")

erreur_standard = sigma / np.sqrt(taille_echantillon)
print(f"Erreur standard de la moyenne (théorique) = {erreur_standard:.2f}")

zn = (moyennes - mu) / (sigma / np.sqrt(taille_echantillon))

plt.figure(figsize=(12, 6))  # Create a figure with a size of 12 inches by 6 inches

# Histogramme des moyennes non normalisées
plt.subplot(1, 2, 1)  # Create a subplot in the first row, first column
plt.hist(moyennes, bins=30, density=True, alpha=0.7, color='blue')  # Plot a histogram of the sample means
plt.axvline(mu, color='red', linestyle='--', label='Moyenne population')  # Add a vertical line at the population mean
plt.title('Distribution des Moyennes des Échantillons')  # Set the title of the plot
plt.xlabel('Moyenne des échantillons')  # Set the x-axis label
plt.ylabel('Densité')  # Set the y-axis label
plt.legend()  # Display the legend

# Histogramme des moyennes normalisées (Zn)
plt.subplot(1, 2, 2)  # Create a subplot in the first row, second column
plt.hist(zn, bins=30, density=True, alpha=0.7, color='green')  # Plot a histogram of the normalized sample means (Z-scores)
x = np.linspace(-4, 4, 100)  # Create an array of 100 evenly spaced points between -4 and 4
phi = (1 / np.sqrt(2 * np.pi)) * np.exp(-x**2 / 2)  # Calculate the probability density function (PDF) of the standard normal distribution
plt.plot(x, phi, color='red', label='N(0,1)')  # Plot the PDF of the standard normal distribution
plt.title('Distribution Normalisée (Zn)')  # Set the title of the plot
plt.xlabel('Zn')  # Set the x-axis label
plt.ylabel('Densité')  # Set the y-axis label
plt.legend()  # Display the legend

plt.tight_layout()  # Adjust the layout of the subplots
plt.show()  # Display the plot


##################################################################################################


import numpy as np  # # pip install numpy pour generer des donnees aleatoires
import pandas as pd  # # pip install pandas pour manipuler les données
import matplotlib.pyplot as plt  # # pip install matplotlib pour visualiser
import seaborn as sns  # # pip install seaborn pour visualiser
from scipy import stats  # # pip install scipy pour calculer des statistiques

np.random.seed(42)  # Set a random seed for reproducibility

n_employes = 100  # Number of employees

# Création des caractéristiques
departements = ['RH', 'IT', 'Marketing', 'Finance', 'Production']
anciennete = ['0-2 ans', '2-5 ans', '5-10 ans', '10+ ans']
niveaux = ['Junior', 'Intermédiaire', 'Senior']

# Génération des données
data = pd.DataFrame({
    'Departement': np.random.choice(departements, n_employes),
    'Anciennete': np.random.choice(anciennete, n_employes),
    'Niveau': np.random.choice(niveaux, n_employes),
    'Satisfaction': np.random.normal(7, 2, n_employes).clip(0, 10)
})

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Distribution par département
sns.countplot(data=data, x='Departement', ax=axes[0, 0])
axes[0, 0].set_title('Distribution par département')
axes[0, 0].tick_params(axis='x', rotation=45)

# Distribution par ancienneté
sns.countplot(data=data, x='Anciennete', ax=axes[0, 1])
axes[0, 1].set_title('Distribution par ancienneté')
axes[0, 1].tick_params(axis='x', rotation=45)

# Distribution par niveau
sns.countplot(data=data, x='Niveau', ax=axes[1, 0])
axes[1, 0].set_title('Distribution par niveau')

# Distribution de la satisfaction
sns.histplot(data=data, x='Satisfaction', ax=axes[1, 1])
axes[1, 1].set_title('Distribution de la satisfaction')

plt.tight_layout()
plt.show()

# Sélection d'un échantillon qualitatif représentatif
echantillon_qualitatif = data.groupby(['Departement', 'Niveau']).apply(
    lambda x: x.sample(n=1) if len(x) > 0 else pd.DataFrame()
).reset_index(drop=True)

print("\nÉchantillon qualitatif représentatif :")
print(echantillon_qualitatif)


##################################################################################################


def calculer_taille_echantillon(marge_erreur, niveau_confiance, proportion=0.5, taille_population=None):
    """
    Calcule la taille d'échantillon nécessaire.

    Args:
        marge_erreur (float): Marge d'erreur souhaitée (ex: 0.05 pour 5%)
        niveau_confiance (float): Niveau de confiance (ex: 0.95 pour 95%)
        proportion (float): Proportion estimée (0.5 si inconnue)
        taille_population (int): Taille de la population totale (None si population infinie)

    Returns:
        int: Taille d'échantillon nécessaire.
    """
    # Z-score pour le niveau de confiance
    z = stats.norm.ppf(1 - (1 - niveau_confiance) / 2)

    # Calcul pour population infinie
    n = (z**2 * proportion * (1 - proportion)) / marge_erreur**2

    # Ajustement pour population finie si spécifiée
    if taille_population is not None:
        n = (n * taille_population) / (n + taille_population - 1)

    return int(np.ceil(n))

# Exemple de calcul
taille_population = 10000
marges_erreur = [0.01, 0.03, 0.05, 0.10]
niveaux_confiance = [0.90, 0.95, 0.99]

resultats = pd.DataFrame(index=marges_erreur, columns=niveaux_confiance)

for me in marges_erreur:
    for nc in niveaux_confiance:
        resultats.loc[me, nc] = calculer_taille_echantillon(
            me, nc, taille_population=taille_population
        )

print("Tailles d'échantillon nécessaires pour différentes marges d'erreur et niveaux de confiance :")
print("Population totale =", taille_population, "\n")
print(resultats)
