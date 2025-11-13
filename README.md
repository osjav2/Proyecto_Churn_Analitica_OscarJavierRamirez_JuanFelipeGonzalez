# 🏦 Proyecto de Predicción de Churn en un Banco

Este proyecto implementa una arquitectura modular (CRISP-DM) para predecir el abandono de clientes (Churn) utilizando modelos de Scikit-learn (optimizados por Recall) y PyCaret (AutoML).

---

## 🚀 Requisitos y Configuración del Entorno

Para ejecutar este proyecto, necesitas tener **Python 3.9 o 3.10** instalado en tu sistema.

### 1. Crear el Entorno Virtual (venv)

Abre la terminal en la carpeta raíz del proyecto y ejecuta los siguientes comandos para crear y activar el entorno virtual:

```bash
# Crear el entorno virtual con Python 3.10 (si está disponible)
python -m venv venv

# Activar el entorno virtual (PowerShell)
.\venv\Scripts\Activate.ps1

2. Instalar Dependencias

Con el entorno (venv) activo, instala todas las librerías necesarias (listadas en requirements.txt):

en la terminal de tu entorno virtual 

pip install -r requirements.txt

el archivo requerimientos contine las librerias usadas en el entorno de instalaciòn se entrega junto con los python


3. Ejecutar el Análisis
El análisis completo (Carga de Datos, EDA, Preprocesamiento, Optimización, Entrenamiento, Evaluación y Guardado de Modelos) se ejecuta desde el cuaderno de Jupyter.

Abre el proyecto en PyCharm o terminal jupyter tener presente el modulo de pycaret solo corre con versiones inferiores a 3.10 de python en colab no correr ya que predetermina version posterior sin embargo el codigo no se detine por este modulo y corre lo demas 

Asigna el intérprete: Asegúrate de que PyCharm use el intérprete (venv) recién creado.

Abre el cuaderno Taller_Final.ipynb.



coloca en el mismo directorio donde esta el cuaderno todos los archivos a ejecutar el venv quedara tambien alli 

├── bank_churn.xlsx
├── Taller_Final.ipynb
├── README.md
├── requirements.txt
├── data_loader.py
├── preprocessing.py
├── model_trainer.py
├── evaluation.py
├── pycaret_automl.py
└── [MODELO_GUARDADO].pkl  (e.g., pycaret_best_automl_model.pkl)
Abre el cuaderno Taller_Final.ipynb.  y ejecuta ya teniendo ejecutaro requermientos y archivos en el mismo directorio


AUTORES: Juan Felipe Gonzalez, Oscar Javier Ramirez

