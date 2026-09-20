# src/DataPreprocessing.py
import numpy as np
import pandas as pd
from src.TextProcessing import TextProcessing


class DataPreprocessing:
    """Preprocesamiento de datos para el clasificador ODS"""

    def __init__(self):
        print("DataPreprocessing.__init__ ->")
        self.text_proc = TextProcessing()
        self.expected_columns = {'textos', 'ods'}
        
        # Mapeo de ODS
        self.ods_map = {
            1: "Fin de la Pobreza",
            2: "Hambre Cero",
            3: "Salud y Bienestar",
            4: "Educación de Calidad",
            5: "Igualdad de Género",
            6: "Agua Limpia y Saneamiento",
            7: "Energía Asequible y No Contaminante",
            8: "Trabajo Decente y Crecimiento Económico",
            9: "Industria, Innovación e Infraestructura",
            10: "Reducción de Desigualdades",
            11: "Ciudades y Comunidades Sostenibles",
            12: "Producción y Consumo Responsable",
            13: "Acción por el Clima",
            14: "Vida Submarina",
            15: "Vida de Ecosistemas Terrestres",
            16: "Paz, Justicia e Instituciones Sólidas",
            17: "Alianzas para Lograr los Objetivos"
        }

    def transform(self, df):
        """
        Pipeline completo de preprocesamiento:
        1. A minúsculas
        2. Eliminar caracteres especiales
        3. Eliminar números
        4. Tokenización y lematización
        """
        print("DataPreprocessing.transform ->")
        
        try:
            df_processed = df.copy()
            
            # Aplicar limpieza básica
            df_processed = self.text_proc.a_minusculas(df_processed)
            df_processed = self.text_proc.eliminar_caracteres_especiales(df_processed)
            df_processed = self.text_proc.eliminar_numeros(df_processed)
            
            # Aplicar preprocesamiento avanzado solo a columna 'textos'
            if 'textos' in df_processed.columns:
                df_processed['textos'] = df_processed['textos'].apply(self.text_proc.text_preprocess)
            print("Preprocesamiento completado")
            return df_processed
            
        except Exception as e:
            print(f"Error en preprocesamiento: {str(e)}")
            return None

    def get_columns(self):
        """Retorna el conjunto de columnas esperadas"""
        print("DataPreprocessing.get_columns ->")
        return self.expected_columns

    def get_categories(self):
        """Retorna lista de nombres de ODS (1-17)"""
        return list(self.ods_map.values())

    def get_ods_name(self, ods_number):
        """Retorna el nombre de un ODS por su número"""
        if isinstance(ods_number, str):
            ods_number = int(ods_number)
        return self.ods_map.get(ods_number, "ODS Desconocido")

    def get_cat_name(self, index):
        """Retorna el nombre de categoría por índice"""
        print("DataPreprocessing.get_cat_name ->")
        categories = self.get_categories()
        if index < 0 or index >= len(categories):
            return ""
        return categories[index]

    def validate_ods(self, ods_value):
        """Valida que el valor ODS esté en el rango 1-17"""
        try:
            ods = int(ods_value)
            return 1 <= ods <= 17
        except:
            return False
    

