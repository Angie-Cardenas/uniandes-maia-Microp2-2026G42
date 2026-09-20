# streamlit_app.py
import Definitions
import pandas as pd
import streamlit as st

from src.ModelController import ModelController


st.set_page_config(
    layout="wide", 
    page_title="ODS Classifier", 
    page_icon="🧑‍🏫",
    initial_sidebar_state="expanded"
)
st.title("🧑‍🏫 Clasificador de Textos por ODS")
st.markdown("**Clasifica textos en los 17 Objetivos de Desarrollo Sostenible (ODS)**")
st.markdown("🔗para más información consulta: https://sdgs.un.org/goals")

@st.cache_resource
def load_controller():
    """Carga el controlador una sola vez"""
    return ModelController()

ctrl = load_controller()

# Aquí creamos una sidebar para seleccionar el modo de configuración, si es simple o usando archivos
st.sidebar.header("⚙️ Configuración")
modo = st.sidebar.radio("Selecciona el modo:",["Predicción Individual", "Predicción usando archivos"])

st.sidebar.markdown("---")
st.sidebar.info(
    "💡 **Información:**\n\n"
    "Este clasificador utiliza:\n"
    "- TF-IDF para vectorización\n"
    "- LSA (20 componentes) para reducción\n"
    "- Random Forest para predicción"
)

# Predicción unitaria
if modo == "Predicción Individual":
    st.header("Predicción Individual")   
    # Área de entrada de texto
    texto_input = st.text_area("Ingresa el texto a clasificar:",placeholder="Escribe aquí un texto sobre desarrollo sostenible...",height=150)  
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔍 Clasificar", use_container_width=True):
            if texto_input.strip():
                with st.spinner("Clasificando..."):
                    resultado = ctrl.predict(texto_input)               
                if resultado['exito']:
                    st.session_state['ultimo_resultado'] = resultado
                else:
                    st.error(f"❌ Error: {resultado['error']}")
            else:
                st.warning("⚠️ Por favor ingresa un texto")
    
    # Mostrar resultado si existe
    if 'ultimo_resultado' in st.session_state:
        resultado = st.session_state['ultimo_resultado']
        
        st.markdown("---")
        st.subheader("📋 Resultado")
        
        # Predicción principal
        col1, col2, col3 = st.columns(3)
        
        with col1:st.metric("ODS Predicho",f"ODS {resultado['ods_predicho']}",border=True)       
        with col2:st.metric("Confianza",f"{resultado['confianza']:.1f}%",border=True)    
        with col3:st.metric("Descripción",resultado['descripcion'],border=True)
        
        # Top 3 predicciones
        st.subheader("🏆 Top 3 posibles alternativas")       
        top_3_df = pd.DataFrame([
            {
                'ODS': f"ODS {pred['ods']}",
                'Descripción': pred['descripcion'],
                'Probabilidad': f"{pred['probabilidad']:.2f}%"
            }
            for pred in resultado['top_3']
        ])       
        st.dataframe(top_3_df, use_container_width=True, hide_index=True)       
        # Texto procesado
        with st.expander("👀 Ver texto ingresado"):
            st.text(resultado['texto_original'])
# Cuando se ingresa un archivo
elif modo == "Predicción usando archivos":
    st.header("Predicción con archivos")   
    uploaded_file = st.file_uploader("Sube un archivo CSV ó excel con una columna que se llame 'textos'",type=["csv", "xlsx"])  
    if uploaded_file is not None:
        try:
            # Cargar datos
            df_input = pd.read_csv(uploaded_file)           
            # Validar estructura
            if 'textos' not in df_input.columns:
                st.error("❌ El CSV debe tener una columna llamada 'textos'")
            else:
                st.success(f"✅ Archivo cargado: {len(df_input)} registros")                
                # Vista previa
                st.subheader("Vista Previa")
                st.dataframe(df_input.head(), use_container_width=True)               
                # Procesar con archivos
                if st.button("Clasificar Todos", use_container_width=True):
                    progress_bar = st.progress(0)
                    status_text = st.empty()                
                    resultados = []
                    for idx, row in df_input.iterrows():
                        # Actualizar progreso
                        progreso = (idx + 1) / len(df_input)
                        progress_bar.progress(progreso)
                        status_text.text(f"Procesando: {idx + 1}/{len(df_input)}")                      
                        # Predicción
                        resultado = ctrl.predict(row['textos'])                       
                        if resultado['exito']:
                            resultados.append({
                                'Índice': idx,
                                'ODS': f"ODS {resultado['ods_predicho']}",
                                'Descripción': resultado['descripcion'],
                                'Confianza': f"{resultado['confianza']:.1f}%"
                            })                  
                    progress_bar.empty()
                    status_text.empty()                    
                    # Mostrar resultados
                    st.subheader("✅ Resultados")
                    df_resultados = pd.DataFrame(resultados)
                    st.dataframe(df_resultados, use_container_width=True, hide_index=True)
                    
                    # Descargar resultados
                    csv = df_resultados.to_csv(index=False)
                    st.download_button( label="⬇️ Descargar Resultados", data=csv,file_name="predicciones_ods.csv", mime="text/csv")
                    
                    # Estadísticas
                    st.subheader("Estadísticas")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Total Procesados", len(df_resultados))
                    
                    with col2:
                        confianza_promedio = df_resultados['Confianza'].str.rstrip('%').astype(float).mean()
                        st.metric("Confianza Promedio", f"{confianza_promedio:.1f}%")
                    
                    with col3:
                        ods_count = df_resultados['ODS'].nunique()
                        st.metric("ODS Únicos Detectados", ods_count)
        
        except Exception as e:
            st.error(f"Ojo! Error al procesar archivo: {str(e)}")
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: gray; font-size: 12px;'>
    ODS Classifier v1.0 | Grupo 42 MLNS <br>
    Basado en Random Forest + LSA + TF-IDF
    </div>
    """,
    unsafe_allow_html=True
)