import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import sigmoid_kernel
import joblib
import os
import requests

class AnimeRecommender:
    def __init__(self):
        self.sig_matrix = None
        self.rec_indices = None
        self.anime_data = None
        self.tfv = None
        
    def download_model(self, file_id, model_path="anime_model.pkl"):
        """
        Descarga el modelo desde Google Drive
        
        Args:
            file_id: ID del archivo en Google Drive
            model_path: Ruta donde guardar el modelo
        """
        if os.path.exists(model_path):
            print(f"📂 Modelo ya existe localmente: {model_path}")
            return True
        
        print("🔄 Descargando modelo desde Google Drive...")
        
        try:
            # URL de descarga directa de Google Drive
            url = f"https://drive.google.com/uc?export=download&id={file_id}"
            
            # Hacer la petición
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            # Guardar archivo
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Mostrar progreso cada 50MB
                        if downloaded % (50 * 1024 * 1024) == 0:
                            progress = (downloaded / total_size) * 100 if total_size > 0 else 0
                            print(f"📥 Descargando... {progress:.1f}%")
            
            print(f"✅ Modelo descargado exitosamente: {model_path}")
            return True
            
        except Exception as e:
            print(f"❌ Error descargando modelo: {e}")
            if os.path.exists(model_path):
                os.remove(model_path)  # Limpiar archivo parcial
            return False
        
    def fit(self, data):
        """
        Entrena el modelo con los datos de anime
        data: DataFrame con columnas ['name', 'genre', 'rating']
        """
        print("🔄 Entrenando modelo de recomendación...")
        
        # 1. Limpiar datos
        self.anime_data = data.copy()
        self.anime_data.drop_duplicates(subset="name", keep="first", inplace=True)
        self.anime_data.reset_index(drop=True, inplace=True)
        
        # 2. Procesar géneros
        genres = self.anime_data["genre"].str.split(", | , | ,").astype(str)
        
        # 3. TF-IDF
        self.tfv = TfidfVectorizer(
            min_df=3, 
            max_features=None, 
            strip_accents="unicode",
            analyzer="word", 
            token_pattern=r"\w{1,}", 
            ngram_range=(1, 3), 
            stop_words="english"
        )
        
        tfv_matrix = self.tfv.fit_transform(genres)
        
        # 4. Calcular similitud (UNA SOLA VEZ)
        print("🧮 Calculando matriz de similitud...")
        self.sig_matrix = sigmoid_kernel(tfv_matrix, tfv_matrix)
        
        # 5. Crear índice de búsqueda
        self.rec_indices = pd.Series(
            self.anime_data.index, 
            index=self.anime_data["name"]
        ).drop_duplicates()
        
        print("✅ Modelo entrenado exitosamente!")
        print(f"📊 Total de animes: {len(self.anime_data)}")
        
    def get_anime_list(self):
        """Retorna lista de todos los animes disponibles"""
        if self.anime_data is None:
            return []
        return sorted(self.anime_data["name"].tolist())
    
    def recommend(self, title, n_recommendations=10):
        """
        Genera recomendaciones para un anime
        
        Args:
            title: Nombre del anime
            n_recommendations: Número de recomendaciones (default: 10)
            
        Returns:
            dict: {"success": bool, "data": list, "message": str}
        """
        try:
            # Validar que el modelo esté entrenado
            if self.sig_matrix is None:
                return {
                    "success": False, 
                    "data": [], 
                    "message": "Modelo no entrenado. Ejecuta fit() primero."
                }
            
            # Validar que el anime existe
            if title not in self.rec_indices:
                available_animes = [name for name in self.rec_indices.index if title.lower() in name.lower()]
                suggestion = f" ¿Quisiste decir: {available_animes[:3]}?" if available_animes else ""
                return {
                    "success": False, 
                    "data": [], 
                    "message": f"Anime '{title}' no encontrado.{suggestion}"
                }
            
            # Obtener índice del anime
            idx = self.rec_indices[title]
            
            # Calcular similitudes
            sig_scores = list(enumerate(self.sig_matrix[idx]))
            sig_scores = sorted(sig_scores, key=lambda x: x[1], reverse=True)
            
            # Obtener top N (excluyendo el mismo anime)
            sig_scores = sig_scores[1:n_recommendations+1]
            anime_indices = [i[0] for i in sig_scores]
            
            # Crear lista de recomendaciones
            recommendations = []
            for i, anime_idx in enumerate(anime_indices):
                rec = {
                    "rank": i + 1,
                    "name": self.anime_data.iloc[anime_idx]["name"],
                    "rating": float(self.anime_data.iloc[anime_idx]["rating"]),
                    "genre": self.anime_data.iloc[anime_idx]["genre"],
                    "similarity_score": round(float(sig_scores[i][1]), 3)
                }
                recommendations.append(rec)
            
            return {
                "success": True,
                "data": recommendations,
                "message": f"Recomendaciones generadas para '{title}'"
            }
            
        except Exception as e:
            return {
                "success": False,
                "data": [],
                "message": f"Error interno: {str(e)}"
            }
    
    def save_model(self, filepath="anime_recommender_model.pkl"):
        """Guarda el modelo entrenado"""
        if self.sig_matrix is None:
            print("❌ No hay modelo entrenado para guardar")
            return False
        
        model_data = {
            'sig_matrix': self.sig_matrix,
            'rec_indices': self.rec_indices,
            'anime_data': self.anime_data,
            'tfv': self.tfv
        }
        
        joblib.dump(model_data, filepath)
        print(f"💾 Modelo guardado en: {filepath}")
        return True
    
    def load_model(self, filepath="anime_model.pkl"):
        """Carga un modelo pre-entrenado"""
        if not os.path.exists(filepath):
            print(f"❌ Archivo no encontrado: {filepath}")
            return False
        
        try:
            model_data = joblib.load(filepath)
            self.sig_matrix = model_data['sig_matrix']
            self.rec_indices = model_data['rec_indices']
            self.anime_data = model_data['anime_data']
            self.tfv = model_data['tfv']
            print(f"📂 Modelo cargado desde: {filepath}")
            return True
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return False

# Ejemplo de uso
if __name__ == "__main__":
    # Cargar tus datos
    # data = pd.read_csv("tu_dataset.csv")
    
    # Crear y entrenar modelo
    recommender = AnimeRecommender()
    # recommender.fit(data)
    
    # Obtener recomendaciones
    # result = recommender.recommend("Naruto", n_recommendations=5)
    # print(result)
    
    # Guardar modelo
    # recommender.save_model()
    
    pass