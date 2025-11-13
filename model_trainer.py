# model_trainer.py

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
import numpy as np


def train_and_optimize_model(model_name, preprocessor, X_train, y_train):
    """
    Define, entrena y optimiza modelos usando RandomizedSearchCV,
    optimizando el Recall (Clase 1: Churn), con hiperparámetros no triviales.
    """

    # Inicialización de variables para evitar UnboundLocalError
    model = None
    n_iter = 0
    param_distributions = {}

    # --- 1. Definición del Modelo y Parámetros ---

    if model_name == 'RandomForest':
        model = RandomForestClassifier(class_weight='balanced', random_state=42)
        param_distributions = {
            'classifier__n_estimators': [100, 200, 400],
            'classifier__max_depth': [8, 15, 25, None],
            'classifier__min_samples_leaf': [2, 4],
            'classifier__max_features': ['sqrt', 0.6, 0.8]
        }
        n_iter = 30

    elif model_name == 'Bagging':
        model = BaggingClassifier(estimator=DecisionTreeClassifier(random_state=42), random_state=42)
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200],
            'classifier__max_samples': [0.5, 0.7, 1.0],
            'classifier__estimator__max_depth': [5, 10, 15],
        }
        n_iter = 20

    elif model_name == 'AdaBoost':
        model = AdaBoostClassifier(estimator=DecisionTreeClassifier(max_depth=1), algorithm='SAMME', random_state=42)
        param_distributions = {
            'classifier__n_estimators': [50, 100, 200, 400],
            'classifier__learning_rate': [0.01, 0.1, 0.5, 1.0],
        }
        n_iter = 20

    elif model_name == 'LogisticRegression':
        model = LogisticRegression(solver='liblinear', class_weight='balanced', random_state=42)
        param_distributions = {'classifier__C': np.logspace(-3, 3, 7)}
        n_iter = 7

    elif model_name == 'SVC':
        model = SVC(probability=True, class_weight='balanced', random_state=42)
        param_distributions = {
            'classifier__C': [0.1, 1, 10],
            'classifier__gamma': ['scale', 'auto'],
            'classifier__kernel': ['rbf'],
        }
        n_iter = 10

    else:
        # Si el nombre del modelo no coincide, lanza un error claro
        raise ValueError(f"Modelo '{model_name}' no soportado en la configuración actual.")

    # --- 2. Creación del Pipeline ---
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])

    # --- 3. Búsqueda y Optimización ---
    print(f"\nIniciando optimización de {model_name}...")

    # Optimizamos el Recall (Clase 1: Churn)
    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=5,
        scoring='recall',  # ¡Métrica de negocio!
        random_state=42,
        n_jobs=-1,
        verbose=0
    )

    search.fit(X_train, y_train)

    print(f"--- Optimización de {model_name} finalizada ---")
    print(f"Mejor Recall (Churn): {search.best_score_:.4f}")
    print(f"Mejores Parámetros:\n {search.best_params_}")

    return search.best_estimator_