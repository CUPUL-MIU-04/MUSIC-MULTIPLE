import os
import tensorflow as tf
import joblib
import json
import numpy as np

class MusicModelLoader:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.genre_labels = None
        self.loaded_model_path = None
    
    def load_latest_model(self, models_dir='models'):
        """Cargar el modelo más reciente entrenado"""
        if not os.path.exists(models_dir):
            raise Exception(f"❌ Directorio de modelos no encontrado: {models_dir}")
        
        # Buscar archivos más recientes
        model_files = [f for f in os.listdir(models_dir) if f.startswith('music_model_') and f.endswith('.h5')]
        
        if not model_files:
            raise Exception("❌ No se encontraron modelos entrenados")
        
        # Ordenar por timestamp (más reciente primero)
        model_files.sort(reverse=True)
        latest_model = model_files[0]
        
        # Extraer timestamp
        timestamp = latest_model.replace('music_model_', '').replace('.h5', '')
        
        # Construir paths
        model_path = os.path.join(models_dir, latest_model)
        scaler_path = os.path.join(models_dir, f'scaler_{timestamp}.pkl')
        metadata_path = os.path.join(models_dir, f'metadata_{timestamp}.json')
        
        # Cargar componentes
        self.model = tf.keras.models.load_model(model_path)
        self.scaler = joblib.load(scaler_path)
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
            self.genre_labels = metadata['genre_labels']
        
        self.loaded_model_path = model_path
        print(f"✅ Modelo cargado: {os.path.basename(model_path)}")
        print(f"🎵 Géneros disponibles: {', '.join(self.genre_labels)}")
        
        return True
    
    def predict_features(self, features):
        """Predecir género a partir de características extraídas"""
        if self.model is None:
            raise Exception("❌ Modelo no cargado")
        
        # Escalar características
        features_scaled = self.scaler.transform([features])
        
        # Predecir
        prediction = self.model.predict(features_scaled, verbose=0)
        predicted_class = np.argmax(prediction[0])
        confidence = prediction[0][predicted_class]
        
        return {
            'genre': self.genre_labels[predicted_class],
            'confidence': float(confidence),
            'all_probabilities': {
                genre: float(prob) for genre, prob in zip(self.genre_labels, prediction[0])
            }
        }

# Instancia global para usar en la API
music_model = MusicModelLoader()
