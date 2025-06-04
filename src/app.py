import streamlit as st
from pickle import load
import numpy as np

# Configuración de página 
st.set_page_config(page_title="Diabetes Prediction App", layout="wide")

# Carga del modelo 
try:
    model_path = "C:/Users/aguil/OneDrive/Desktop/proyectos javier/ML-WEBAPP-USING-STREAMLIT/data/models/Random_forest_model.sav"
    with open(model_path, 'rb') as f:
        model = load(f)
except Exception as e:
    st.error(f"Error loading model: {str(e)}")
    st.stop()

# Interfaz de usuario
st.title('Diabetes Prediction Tool')

# Organización en columnas
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.slider("Pregnancies", 0, 17, 1)
    Glucose = st.slider("Glucose", 0, 200, 100)
    BloodPressure = st.slider("Blood Pressure", 20, 120, 70)
    BMI = st.slider("BMI", 10.0, 70.0, 25.0, 0.5)

with col2:
    Insulin = st.number_input("Insulin level", 0, 600, 80)
    DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", 0.0, 2.5, 0.5, 0.01)
    Age = st.number_input("Age", 1, 120, 30)

# Botón de predicción
if st.button("Predict Diabetes Risk"):
    try:
        # Convertir inputs a array numpy
        input_data = np.array([
            [Pregnancies, Glucose, BloodPressure, Insulin, BMI, 
             DiabetesPedigreeFunction, Age]
        ])
        
        # Realizar predicción
        prediction = model.predict(input_data)[0]
        result = "YES (High Risk)" if prediction == 1 else "NO (Low Risk)"
        
        # Mostrar resultado con estilo
        st.subheader("Prediction Result")
        if prediction == 1:
            st.error(result)
        else:
            st.success(result)
            
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")