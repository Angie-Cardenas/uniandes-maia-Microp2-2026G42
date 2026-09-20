# src/TextProcessing.py
import pandas as pd
import pandas.api.types as ptypes
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

class TextProcessing:
    """Funciones compartidas de preprocesamiento"""
    
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = RegexpTokenizer(r'\w+')

    @staticmethod
    def remover_tildes(texto):
        """Elimina tildes"""
        if isinstance(texto, str):
            tabla_acentos = str.maketrans('áéíóúüÁÉÍÓÚÜ', 'aeiouuAEIOUU')
            return texto.translate(tabla_acentos)
        return texto

    @staticmethod
    def a_minusculas(df):
        """Convierte a minúsculas"""
        df_copy = df.copy()
        for col in df_copy.columns:
            if ptypes.is_string_dtype(df_copy[col]) or ptypes.is_object_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str).str.lower()
        return df_copy

    @staticmethod
    def eliminar_caracteres_especiales(df):
        """Elimina caracteres especiales"""
        df_copy = df.copy()
        for col in df_copy.columns:
            if ptypes.is_string_dtype(df_copy[col]) or ptypes.is_object_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str).str.replace(r'[^a-z0-9ñ\s]', ' ', regex=True)
                df_copy[col] = df_copy[col].str.strip().str.replace(r'\s+', ' ', regex=True)
        return df_copy

    @staticmethod
    def eliminar_numeros(df):
        """Elimina números"""
        df_copy = df.copy()
        for col in df_copy.columns:
            if ptypes.is_string_dtype(df_copy[col]) or ptypes.is_object_dtype(df_copy[col]):
                df_copy[col] = df_copy[col].astype(str).str.replace(r'\d', ' ', regex=True)
                df_copy[col] = df_copy[col].str.strip().str.replace(r'\s+', ' ', regex=True)
        return df_copy

    def text_preprocess(self, text):
        """Tokenización, lematización y stopwords"""
        if not isinstance(text, str):
            return ""
        tokens = self.tokenizer.tokenize(text)
        tokens = [self.remover_tildes(word) for word in tokens if word not in stopwords.words('spanish')]
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        tokens = [word for word in tokens if len(word) >= 3]
        return ' '.join(tokens)