import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import make_scorer, average_precision_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from code.param import *

def vect_split_data(df):
    """ Divise les données en ensembles d'entraînement et de validation. """
    X = df['ecfp'].tolist()  # Convertir la colonne 'ecfp' en liste de listes
    y = df['binds']  # La colonne cible
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.3, random_state=RANDOM_STATE)
    return X_train, X_val, y_train, y_val

def vect_train_and_evaluate(X_train, X_val, y_train, y_val):
    """ Entraîne plusieurs modèles en utilisant GridSearch et évalue leurs performances. """
    param_grid = {
        'Logistic Regression': {
            'model': LogisticRegression(max_iter=1000, solver='lbfgs'),
            'params': {'C': [0.1, 1, 10]}
        },
        'Random Forest': {
            'model': RandomForestClassifier(),
            'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
        },
        'Support Vector Machine': {
            'model': SVC(probability=True),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        }
    }

    average_precision_scorer = make_scorer(average_precision_score, response_method='predict_proba')
    best_model = None
    best_score = 0

    for name, mp in param_grid.items():
        grid = GridSearchCV(mp['model'], mp['params'], cv=5, scoring=average_precision_scorer)
        grid.fit(X_train, y_train)

        # Meilleur modèle
        model = grid.best_estimator_
        y_proba = model.predict_proba(X_val)[:, 1]
        ap_score = average_precision_score(y_val, y_proba)

        print('\n------------------------------------------------------------')
        print(f"{name}:")
        print(f"  - Best Average Precision: {ap_score:.4f}")
        print(f"  - Best Parameters: {grid.best_params_}")

        if ap_score > best_score:
            best_score = ap_score
            best_model = model

        y_pred = model.predict(X_val)
        cm = confusion_matrix(y_val, y_pred)
        # plt.figure(figsize=(6, 4))
        # sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])
        # plt.title(f"Matrice de Confusion - {name}")
        # plt.xlabel("Predicted")
        # plt.ylabel("Actual")
        # plt.show()

        print(classification_report(y_val, y_pred))
        print('------------------------------------------------------------')

    return best_model

def vect_save_model(model, name_protein, nb_sample):
    """ Enregistre le meilleur modèle trouvé. """
    parent_dir = os.path.dirname(os.getcwd())
    model_path = os.path.join(parent_dir, f'drug_smile/raw_data/best_model_vector_SVC_{name_protein}_{nb_sample}.pkl\n')
    joblib.dump(model, model_path)
    print(f"\nModèle enregistré à : {model_path}")
