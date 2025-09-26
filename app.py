import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de Citas Médicas",
    page_icon="🏥",
    layout="wide",
)

# --- Funciones de Caching para Cargar Datos y Modelo ---
@st.cache_data
def load_data(file_path):
    """
    Carga y preprocesa los datos replicando los pasos del notebook de Colab.
    """
    df = pd.read_csv(file_path)
    
    # Replicamos los mismos pasos de tu notebook
    df.rename(columns={'Hipertension': 'Hypertension', 
                       'Handcap': 'Handicap',
                       'No-show': 'NoShow',
                       'ScheduledDay': 'ScheduledDate',
                       'AppointmentDay': 'AppointmentDate'}, inplace=True)

    # Convertir a datetime y manejar errores
    df['ScheduledDate'] = pd.to_datetime(df['ScheduledDate'], errors='coerce')
    df['AppointmentDate'] = pd.to_datetime(df['AppointmentDate'], errors='coerce')
    df.dropna(subset=['ScheduledDate', 'AppointmentDate'], inplace=True)

    # Crear la característica 'LeadTime'
    df['LeadTime'] = (df['AppointmentDate'] - df['ScheduledDate']).dt.days

    # Limpiar datos inconsistentes
    df = df[df['LeadTime'] >= 0]
    df = df[df['Age'] >= 0]
    
    return df

@st.cache_resource
def load_pipeline(pipeline_path):
    """Carga el pipeline de machine learning entrenado."""
    return joblib.load(pipeline_path)

# --- Carga de Datos y Modelo ---
df = load_data('KaggleV2-May-2016.csv')
pipeline = load_pipeline('medical_appointment_pipeline.pkl')

# --- Barra Lateral de Navegación ---
st.sidebar.title("Navegación")
page = st.sidebar.radio("Ir a", ["Análisis Exploratorio de Datos", "Realizar una Predicción"])

# --- PÁGINA DE ANÁLISIS EXPLORATORIO (EDA) ---

if page == "Análisis Exploratorio de Datos":
    st.title("📊 Análisis Exploratorio de Datos (EDA)")
    st.markdown("Explora la relación entre las características del paciente y el ausentismo a las citas.")

    # Métricas Clave (KPIs) 
    st.header("Indicadores Clave de Rendimiento (KPIs)")
    total_appointments = len(df)
    no_show_count = df['NoShow'].value_counts().get('Yes', 0)
    no_show_rate = (no_show_count / total_appointments) * 100
    avg_lead_time = df['LeadTime'].mean()

    col1, col2, col3 = st.columns(3)
    col1.metric("Total de Citas", f"{total_appointments:,}")
    col2.metric("Tasa de Ausentismo", f"{no_show_rate:.2f}%")
    col3.metric("Tiempo de Espera Promedio", f"{avg_lead_time:.1f} días")

    st.markdown("---")

    # Visualizaciones Interactivas 
    st.header("Visualizaciones Interactivas")

    # Visualización 1: Tasa de Ausentismo por Característica
    st.subheader("Tasa de Ausentismo por Característica del Paciente")
    cat_feature = st.selectbox(
        "Selecciona una característica para analizar:",
        ['Gender', 'Scholarship', 'Hypertension', 'Diabetes', 'Alcoholism', 'SMS_received']
    )
    
    no_show_by_feature = df.groupby(cat_feature)['NoShow'].value_counts(normalize=True).unstack().fillna(0)
    no_show_rate_by_feature = no_show_by_feature.get('Yes', pd.Series(0, index=no_show_by_feature.index)) * 100
    
    fig1 = px.bar(
        no_show_rate_by_feature,
        x=no_show_rate_by_feature.index,
        y=no_show_rate_by_feature.values,
        title=f"Tasa de Ausentismo por {cat_feature}",
        labels={'x': cat_feature, 'y': 'Tasa de Ausentismo (%)'},
        color_discrete_sequence=px.colors.qualitative.Pastel
    )
    st.plotly_chart(fig1, use_container_width=True)
    
    st.info(
        f"""
        **💡 ¿Qué podemos entender de esta gráfica?**
        
        Esta gráfica de barras muestra cómo cambia la tasa de ausentismo (No-Show) según la característica que elijas. 
        
        - **Observa las diferencias en altura:** Si una barra es significativamente más alta que otra, indica que ese grupo de pacientes tiende a faltar más a sus citas.
        - **Ejemplo:** Al seleccionar `SMS_received`, puedes notar un hecho contraintuitivo: los pacientes que recibieron un SMS tienen una tasa de ausentismo mayor. Esto podría deberse a que los recordatorios se envían para citas agendadas con mucha antelación, que son las que más se suelen olvidar.
        """
    )


    # Visualización 2: Distribución de Edad vs. Ausentismo
    st.subheader("Distribución de Edad vs. Estado de la Cita")
    age_slider = st.slider("Selecciona un rango de edad:", 0, 115, (0, 100))
    filtered_df_age = df[(df['Age'] >= age_slider[0]) & (df['Age'] <= age_slider[1])]
    
    fig2 = px.histogram(
        filtered_df_age, x='Age', color='NoShow', barmode='overlay',
        title="Distribución de Edad por Estado de la Cita",
        labels={'Age': 'Edad del Paciente', 'NoShow': 'Estado de la Cita'},
        color_discrete_map={'Yes': 'orangered', 'No': 'cornflowerblue'}
    )
    st.plotly_chart(fig2, use_container_width=True)

    st.info(
        """
        **💡 ¿Qué podemos entender de esta gráfica?**

        Este histograma te permite ver qué grupos de edad tienen más citas y cómo se compara su comportamiento de asistencia.
        
        - **Picos en la distribución:** Los picos más altos muestran los rangos de edad con mayor cantidad de citas. Generalmente, hay picos en niños pequeños y en adultos.
        - **Comparación de colores:** Dentro de cada barra de edad, puedes ver la proporción de pacientes que asistieron (azul) frente a los que no (naranja). Esto ayuda a identificar si un grupo de edad específico tiene una tendencia mayor a faltar.
        """
    )
    
    # Visualización 3: Lead Time vs. No-Show
    st.subheader("Impacto del Tiempo de Espera en el Ausentismo")
    fig3 = px.box(
        df, x='NoShow', y='LeadTime', color='NoShow',
        title="Distribución del Tiempo de Espera para Asistentes vs. Ausentes",
        labels={'NoShow': 'Estado de la Cita', 'LeadTime': 'Días de Espera (Lead Time)'},
        color_discrete_map={'Yes': 'orangered', 'No': 'cornflowerblue'}
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.info(
        """
        **💡 ¿Qué podemos entender de esta gráfica?**

        Esta es una de las visualizaciones más importantes. Un diagrama de caja (boxplot) muestra la distribución de un dato.

        - **La clave está en la mediana (la línea dentro de la caja):** La caja naranja ("Yes", No-Show) está mucho más arriba que la azul ("No", Show-Up).
        - **Conclusión principal:** Los pacientes que **no se presentan** a sus citas tienen, en promedio, un **tiempo de espera (Lead Time) mucho mayor** que los que sí asisten. Agendar una cita con demasiada antelación es un fuerte predictor de ausentismo.
        """
    )

# --- PÁGINA DE PREDICCIÓN ---

elif page == "Realizar una Predicción":
    st.title("🔮 Predecir Ausentismo de Pacientes")
    st.markdown("Ingresa los detalles del paciente para predecir la probabilidad de que no se presente.")

    neighbourhoods = sorted(df['Neighbourhood'].unique())

    with st.form(key='prediction_form'):
        st.header("Información del Paciente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad", 0, 120, 30, 1)
            gender = st.selectbox("Género", ["M", "F"])
            lead_time = st.number_input("Días de Espera (Lead Time)", 0, 365, 5, 1, help="Número de días entre que se agenda y la fecha de la cita.")
            neighbourhood = st.selectbox("Barrio (según datos originales de Brasil)", options=neighbourhoods)

        with col2:
            scholarship = st.checkbox("Tiene Beca (Bolsa Família)")
            hypertension = st.checkbox("Tiene Hipertensión")
            diabetes = st.checkbox("Tiene Diabetes")
            alcoholism = st.checkbox("Sufre de Alcoholismo")
            handicap = st.checkbox("Tiene alguna Discapacidad")
            sms_received = st.checkbox("Recibió un SMS de notificación")

        submit_button = st.form_submit_button(label="Predecir Ausentismo")

    if submit_button:
        # Crear un DataFrame con los datos del formulario
        input_data = {
            'Age': [age], 'Gender': [gender], 'Scholarship': [int(scholarship)],
            'Hypertension': [int(hypertension)], 'Diabetes': [int(diabetes)],
            'Alcoholism': [int(alcoholism)], 'Handicap': [int(handicap)],
            'SMS_received': [int(sms_received)], 'LeadTime': [lead_time],
            'Neighbourhood': [neighbourhood]
        }
        input_df = pd.DataFrame(input_data)

        # Realizar la predicción
        prediction = pipeline.predict(input_df)[0]
        prediction_proba = pipeline.predict_proba(input_df)[0]
        
        st.subheader("Resultado de la Predicción")
        if prediction == 1: # 1 significa 'No-Show'
            st.error("Predicción: El paciente PROBABLEMENTE NO SE PRESENTARÁ.", icon="🚨")
            st.write(f"Confianza de la predicción: **{prediction_proba[1]*100:.2f}%**")
        else:
            st.success("Predicción: El paciente PROBABLEMENTE SÍ SE PRESENTARÁ.", icon="✅")
            st.write(f"Confianza de la predicción: **{prediction_proba[0]*100:.2f}%**")
        
        # --- Explicación de la predicción ---
        st.subheader("Factores Clave en esta Predicción")
        explanation = []
        if lead_time > 15:
            explanation.append(f"- El **tiempo de espera de {lead_time} días** es largo, lo que aumenta significativamente la probabilidad de ausentismo.")
        elif lead_time < 3:
             explanation.append(f"- El **tiempo de espera de {lead_time} días** es corto, lo que favorece la asistencia del paciente.")

        if sms_received and lead_time > 7:
            explanation.append("- Haber **recibido un SMS** para una cita lejana a veces se correlaciona con un mayor riesgo de no presentarse, ya que estas citas son más fáciles de olvidar o cancelar.")
        
        if hypertension or diabetes:
            explanation.append("- Pacientes con **condiciones crónicas** como hipertensión o diabetes suelen tener un seguimiento médico más constante, lo que podría influir en su asistencia.")

        if not explanation:
            explanation.append("- Los factores de este paciente no muestran una clara tendencia hacia el ausentismo o la asistencia, según las reglas más comunes.")

        st.info("**¿En qué se basa el modelo para esta conclusión?**\n" + "\n".join(explanation))
