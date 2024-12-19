# drug_smile

## Description

**drug_smile** est une librairie Python permettant de prédire les interactions entre des molécules et des protéines à l'aide de représentations SMILES et de techniques de Machine Learning et de Deep Learning. Ce projet a été réalisé dans le cadre d'une formation "Data Science & IA" organisée par Le Wagon.

Il s'appuie sur le projet Kaggle **[NeurIPS 2024 - Predict New Medicines with BELKA](https://www.kaggle.com/competitions/leash-BELKA/overview)**.

Cette librairie a été développée dans le cadre d'un projet collaboratif par :
- Lucas Sedran ([lucas.sedran@hotmail.fr](mailto:lucas.sedran@hotmail.fr)) ([https://www.linkedin.com/in/lucassedran/](https://www.linkedin.com/in/lucassedran/))
- Benoit Cochet
- Dorian Schnepp
- Issam Mehnana

Les principales fonctionnalités incluent :
- `drug_smile/_00_prepration` : La création des échantillons.
- `drug_smile/_01_preprocessing` : La transformation des molécules du format SMILES en formats exploitables : **caractéristiques chimiques** (appelé '*cara*'), **ECFP** (appelé '*vect*') et **graphes** (appelé '*GNN*').
- `drug_smile/_02_model_train` : L'entraînement de modèles d'apprentissage : **SVC** (pour les caractéristiques chimiques et les ECFP), **Régression Logistique** (pour les ECFP) et **GNN** (pour les graphes).
- `drug_smile/_03_predict` : La prédiction d'interactions molécule-protéine.

## Installation

### Prérequis

Assurez-vous que **Python 3.8+** est installé sur votre machine. Cette librairie a été développée et testée sur **Python 3.10.6**.

### Étapes d'installation

**1. Clonez le dépôt GitHub :**

   ```bash
   git clone https://github.com/lucas-sedran/drug_smile.git
   cd drug_smile
  ```

**2. Créez un environnement virtuel (optionnel mais recommandé) :**

Soit avec venv (rapide et simple) :
  ```bash
  python -m venv drug_smile-env
  source drug_smile-env/bin/activate # Sur macOS/Linux
  drug_smile-env\Scripts\activate    # Sur Windows
  ```

Soit avec pyenv (si vous devez gérer plusieurs versions de Python) :
  ```bash
  pyenv install 3.10.6
  pyenv virtualenv 3.10.6 drug_smile-env
  pyenv local drug_smile-env
  ```
**3. Configurez les variables d'environnement :**

Certaines fonctionnalités de la librairie nécessitent des variables d'environnement. Suivez ces étapes pour les configurer :

Copiez le fichier **`.env.sample`** et renommez-le en **`.env`** :
  ```bash
  cp .env.sample .env
  ```

Installez et configurez **`direnv`** pour gérer automatiquement les variables d'environnement :
1. Installez **`direnv`** :
  ```bash
  brew install direnv    # Sur macOS
  sudo apt install direnv # Sur /Linux
  ```
Sur Windows : Utilisez **WSL** pour installer direnv ou gérez les variables avec un autre outil.

2. Ajoutez **direnv** à votre shell (bash, zsh, etc.) en suivant les instructions [officielles](https://direnv.net/docs/hook.html).

3. Autorisez l'utilisation des variables d'environnement dans le répertoire du projet :
  ```bash
  direnv allow
  ```
4. En cas de mise à jour des variables d'environnement :
  ```bash
  direnv reload
  ```

**4. Installez la librairie et ses dépendances :**

Avec `setup.py` :
  ```bash
  pip install .
  ```
  Si vous rencontrez des problèmes, installez les dépendances directement depuis `requirements.txt` :
  ```bash
  pip install -r requirements.txt
  ```

## Utilisation
### Notebook interactif
Utilisez le notebook `main.ipynb` pour suivre un exemple complet d'utilisation de la librairie.

### Pipelines automatisées
Les **pipelines** disponibles dans le `Makefile` permettent d'automatiser plusieurs étapes du projet, notamment :
- La préparation des données
- Le pré-traitement des données
- L'entraînement de modèles
- La prédiction

## Licence
Ce projet est distribué sans **licence explicite**. Contactez les auteurs pour plus d'informations.

## Liens utiles
Projet Kaggle : https://www.kaggle.com/competitions/leash-BELKA/overview

Dépôt GitHub principal : https://github.com/lucas-sedran/drug_smile
