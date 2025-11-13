# preprocessing.py
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer # <-- Importante para manejo de nulos

def create_preprocessor(X_train):
    """
    Crea un ColumnTransformer para preprocesar los datos numéricos y categóricos.
    Incluye imputación, escalado y One-Hot Encoding dentro del pipeline.
    """

    categorical_features = X_train.select_dtypes(include=['object']).columns
    numerical_features = X_train.select_dtypes(exclude=['object']).columns # Asume int/float

    # Pipeline para transformaciones numéricas: Imputación (Media) + Escalado
    numeric_transformer = Pipeline(steps=[
        # Maneja NaNs en variables numéricas
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())
    ])

    # Pipeline para transformaciones categóricas: Imputación (Moda) + OHE
    categorical_transformer = Pipeline(steps=[
        # Maneja NaNs en variables categóricas
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # sparse_output=False es útil para Random Forest
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Combina las transformaciones
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        # 'remainder='drop'' elimina las columnas no especificadas (ej: 'clientnum')
        remainder='drop'
    )

    print("Preprocessor creado exitosamente (Incluye Imputación, Escalado, y OHE).")
    return preprocessor