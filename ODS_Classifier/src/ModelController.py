# src/ModelController.py
import Definitions
import numpy as np
import os.path as osp
import pandas as pd
from io import StringIO
import joblib

from src.DataPreprocessing import DataPreprocessing
from src.TextProcessing import TextProcessing


class ModelController:

    def __init__(self):
        print("ModelController.__init__ ->")
        
        # Ruta base de modelos
        self.model_path = osp.join(Definitions.ROOT_DIR, "resources/models")
        
        # Rutas de cada artefacto
        self.tfidf_path = osp.join(self.model_path, "tfidf_vectorizer.joblib")
        self.svd_path = osp.join(self.model_path, "svd_lsa.joblib")
        self.model_file = osp.join(self.model_path, "model.joblib")

        # Cargar modelos desde joblib
        try:
            self.tfidf = joblib.load(self.tfidf_path)
            self.svd_lsa = joblib.load(self.svd_path)
            self.model = joblib.load(self.model_file)
            print("Modelos cargados exitosamente")
        except Exception as e:
            print(f"Error al cargar modelos: {str(e)}")
            raise

        # Inicializar preprocesamiento
        self.text_proc = TextProcessing()
        self.d_processing = DataPreprocessing()
        
        # Mapa de ODS
        self.ods_map = self.d_processing.ods_map
        
        # DataFrame de entrada
        self.input_df = None

    def validate_data(self, df):
        """Compara los nombres de las columnas con lo esperado"""
        return self.d_processing.get_columns().issubset(set(df.columns))
    
    def get_categories(self):
        """Retorna las categorías de ODS"""
        print("ModelController.get_categories ->")
        return self.d_processing.get_categories()    

    def load_input_data(self, input_data):
        """Carga datos de entrada desde un archivo CSV"""
        print("ModelController.load_input_data ->")
        try:
            input_data_str = StringIO(input_data.getvalue().decode("utf-8"))
            self.input_df = pd.read_csv(input_data_str)
            is_valid = self.validate_data(self.input_df)
            print(f"Datos cargados: {len(self.input_df)} registros")
            return self.input_df, is_valid

        except Exception as e:
            raise Exception(f"Error al leer datos de entrada: {str(e)}")

    def predict(self, text_data):
        """
        Realiza predicción de ODS para un texto     
        Args:text_data: String con el texto a clasificar
        Returns:dict con predicción, confianza y top 3 alternativas
        """
        print("ModelController.predict ->")
        
        try:
            # Obtener texto de entrada
            if isinstance(text_data, str):
                text_input = text_data
            else:
                text_input = str(text_data)
            
            # Paso 1: Limpieza básica
            df_temp = pd.DataFrame({'textos': [text_input]})
            df_cleaned = self.text_proc.a_minusculas(df_temp)
            df_cleaned = self.text_proc.eliminar_caracteres_especiales(df_cleaned)
            df_cleaned = self.text_proc.eliminar_numeros(df_cleaned)
            
            # Paso 2: Preprocesamiento avanzado (tokenización, lematización)
            text_preprocessed = self.text_proc.text_preprocess(df_cleaned['textos'].iloc[0])
            # Paso 3: Vectorización TF-IDF
            tfidf_vector = self.tfidf.transform([text_preprocessed]) 
            # Paso 4: Reducción de dimensionalidad (LSA)
            X_reduced = self.svd_lsa.transform(tfidf_vector)
            # Paso 5: Predicción
            y_pred = self.model.predict(X_reduced)[0]
            probabilities = self.model.predict_proba(X_reduced)[0]
            confidence = max(probabilities) * 100           
            # Paso 6: Top 3 predicciones
            top_3_idx = (-probabilities).argsort()[:3]
            top_3_predictions = [
                {
                    'ods': int(self.model.classes_[idx]),
                    'descripcion': self.ods_map.get(int(self.model.classes_[idx]), 'Desconocido'),
                    'probabilidad': float(probabilities[idx] * 100)
                }
                for idx in top_3_idx
            ]
            
            # Retornar resultado completo
            result = {
                'exito': True,
                'ods_predicho': int(y_pred),
                'descripcion': self.ods_map.get(int(y_pred), 'Desconocido'),
                'confianza': float(confidence),
                'top_3': top_3_predictions,
                'texto_original': text_input[:200] + '...' if len(text_input) > 200 else text_input
            }
            
            return result

        except Exception as e:
            return {
                'exito': False,
                'error': f"Error en predicción: {str(e)}"
            }

    def predict_batch(self, df):
        """
        Realiza predicciones en lote para un DataFrame        
        Args:df: DataFrame con columna 'textos'          
        Returns:DataFrame con predicciones
        """
        print("ModelController.predict_batch ->")
        
        results = []
        for idx, row in df.iterrows():
            result = self.predict(row['textos'])
            result['indice'] = idx
            results.append(result)
        
        return pd.DataFrame(results)


