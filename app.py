import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np

# Cargar dataset real
@st.cache_data
def cargar_datos():
    url = "https://raw.githubusercontent.com/Zephyrodes/Datasets/main/diabetes_binary_5050split_health_indicators_BRFSS2021.csv"
    data = pd.read_csv(url)
    X = data.drop("Diabetes_binary", axis=1)
    y = data["Diabetes_binary"]
    return X, y

# Datos y modelo
X, y = cargar_datos()
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

modelo = RandomForestClassifier(n_estimators=100, random_state=42)
modelo.fit(X_train, y_train)

# Interfaz
st.set_page_config(page_title="Predicción de Diabetes BRFSS", page_icon="🧬")
st.title("🩺 Sistema Predictivo de Diabetes - BRFSS Dataset")

st.sidebar.header("Ingresa tus datos de salud")

# Campos disponibles en el dataset
campos_formulario = {
    "HighBP": st.sidebar.radio("Presión alta (HighBP)", [0, 1]),
    "HighChol": st.sidebar.radio("Colesterol alto (HighChol)", [0, 1]),
    "CholCheck": st.sidebar.radio("Chequeo de colesterol (CholCheck)", [0, 1]),
    "BMI": st.sidebar.slider("IMC (BMI)", 10, 60, 25),
    "Smoker": st.sidebar.radio("¿Fuma? (Smoker)", [0, 1]),
    "Stroke": st.sidebar.radio("¿Ha tenido un derrame cerebral? (Stroke)", [0, 1]),
    "HeartDiseaseorAttack": st.sidebar.radio("Enfermedad cardiaca (HeartDiseaseorAttack)", [0, 1]),
    "PhysActivity": st.sidebar.radio("Actividad física (PhysActivity)", [0, 1]),
    "Fruits": st.sidebar.radio("¿Consume frutas? (Fruits)", [0, 1]),
    "Veggies": st.sidebar.radio("¿Consume verduras? (Veggies)", [0, 1]),
    "HvyAlcoholConsump": st.sidebar.radio("¿Alcohólico pesado? (HvyAlcoholConsump)", [0, 1]),
    "AnyHealthcare": st.sidebar.radio("¿Tiene algún seguro médico? (AnyHealthcare)", [0, 1]),
    "NoDocbcCost": st.sidebar.radio("¿Evita doctor por costo? (NoDocbcCost)", [0, 1]),
    "GenHlth": st.sidebar.slider("Salud general (GenHlth)", 1, 5, 3),
    "MentHlth": st.sidebar.slider("Días de salud mental mala (MentHlth)", 0, 30, 5),
    "PhysHlth": st.sidebar.slider("Días de salud física mala (PhysHlth)", 0, 30, 5),
    "DiffWalk": st.sidebar.radio("¿Tiene dificultad para caminar? (DiffWalk)", [0, 1]),
    "Sex": st.sidebar.radio("Sexo", [0, 1]),
    "Age": st.sidebar.slider("Grupo de edad codificado (Age)", 1, 13, 7),
    "Education": st.sidebar.slider("Nivel educativo codificado (Education)", 1, 6, 3),
    "Income": st.sidebar.slider("Nivel de ingreso codificado (Income)", 1, 8, 4)
}

# Convertir entrada a DataFrame
entrada_usuario = pd.DataFrame([campos_formulario])

# Escalado y predicción
entrada_escalada = scaler.transform(entrada_usuario)
prediccion = modelo.predict(entrada_escalada)[0]
probabilidad = modelo.predict_proba(entrada_escalada)[0][1]

# Resultado
st.subheader("🧪 Resultado de la Predicción")
st.write("**Datos ingresados:**")
st.table(entrada_usuario)

if prediccion == 1:
    st.error(f"🔴 Riesgo detectado de diabetes (probabilidad: {probabilidad:.2f})")
    if probabilidad > 0.7:
        st.markdown("### 🚨 Acción sugerida: Contactar inmediatamente con un profesional médico.")
    else:
        st.markdown("### ⚠️ Acción sugerida: Realizar chequeo médico preventivo.")
else:
    st.success(f"🟢 Sin indicios de diabetes (probabilidad: {probabilidad:.2f})")
    st.markdown("### ✅ Acción sugerida: Mantener hábitos saludables y control periódico.") 
