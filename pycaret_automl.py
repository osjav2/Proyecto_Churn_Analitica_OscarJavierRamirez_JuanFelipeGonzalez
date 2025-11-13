# pycaret_automl.py
import pandas as pd
from pycaret.classification import setup, compare_models, tune_model, finalize_model, save_model, pull


def run_pycaret_automl(df, target_column='attrition_flag'):
    # ... (Se omite la implementación para brevedad, usando tu código original) ...

    model_filename = 'pycaret_best_automl_model'

    print("\n--- 5. PyCaret: Inicializando entorno y comparando modelos ---")

    # 5.1 Configuración del entorno
    s = setup(
        data=df,
        target=target_column,
        session_id=42,
        html=False
    )

    # 5.2 Comparación de modelos (Optimizado por Recall)
    best_pycaret_model = compare_models(
        sort='Recall',  # <-- Optimiza la métrica de negocio
        n_select=1,
        exclude=['ridge', 'dummy', 'svm', 'quadratic_discriminant_analysis']
    )

    # Imprimir tabla de comparación de PyCaret (para el informe)
    print("\nTabla de comparación de modelos de PyCaret (Optimizado por Recall):")
    display(pull())

    # 5.3 Ajuste Fino del Mejor Modelo (Tuning)
    tuned_model_pycaret = tune_model(
        best_pycaret_model,
        optimize='Recall',
        n_iter=30
    )

    # 5.4 Finalización y Guardado
    final_pycaret_model = finalize_model(tuned_model_pycaret)
    save_model(final_pycaret_model, model_filename)

    return model_filename