import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
import joblib
import librosa
import json
from datetime import datetime

class MusicModelTrainer:
    def __init__(self):
        self.model = None
        self.scaler = None
        self.genre_labels = ['rock', 'jazz', 'classical', 'pop', 'hiphop', 'electronic']
        
    def extract_features(self, audio_path):
        """Extraer características de audio usando librosa"""
        try:
            y, sr = librosa.load(audio_path, duration=30)  # 30 segundos
            
            features = []
            
            # MFCCs (Mel-frequency cepstral coefficients)
            mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
            mfcc_mean = np.mean(mfcc, axis=1)
            features.extend(mfcc_mean)
            
            # Spectral features
            spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=y, sr=sr))
            spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr))
            spectral_contrast = np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1)
            
            features.append(spectral_centroid)
            features.append(spectral_rolloff)
            features.extend(spectral_contrast)
            
            # Rhythm features
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
            features.append(tempo)
            
            # Zero crossing rate
            zcr = np.mean(librosa.feature.zero_crossing_rate(y))
            features.append(zcr)
            
            return np.array(features)
            
        except Exception as e:
            print(f"Error procesando {audio_path}: {e}")
            return None
    
    def create_model(self, input_dim, num_classes):
        """Crear modelo de red neuronal"""
        model = keras.Sequential([
            keras.layers.Dense(256, activation='relu', input_shape=(input_dim,)),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(num_classes, activation='softmax')
        ])
        
        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        
        return model
    
    def train_from_directory(self, data_dir, epochs=50, test_size=0.2):
        """Entrenar modelo desde directorio de audios organizado por géneros"""
        features = []
        labels = []
        
        print("🔊 Extrayendo características de audio...")
        
        for genre_idx, genre in enumerate(self.genre_labels):
            genre_dir = os.path.join(data_dir, genre)
            
            if not os.path.exists(genre_dir):
                print(f"⚠️  Directorio {genre_dir} no encontrado, saltando...")
                continue
                
            for audio_file in os.listdir(genre_dir):
                if audio_file.endswith(('.wav', '.mp3', '.flac')):
                    audio_path = os.path.join(genre_dir, audio_file)
                    feature = self.extract_features(audio_path)
                    
                    if feature is not None:
                        features.append(feature)
                        labels.append(genre_idx)
                        print(f"✅ Procesado: {audio_file} -> {genre}")
        
        if len(features) == 0:
            raise Exception("❌ No se encontraron archivos de audio para entrenar")
        
        # Convertir a arrays numpy
        X = np.array(features)
        y = np.array(labels)
        
        print(f"📊 Dataset: {X.shape[0]} muestras, {X.shape[1]} características")
        
        # Normalizar características
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = train_test_split(
            X_scaled, y, test_size=test_size, random_state=42, stratify=y
        )
        
        # Crear y entrenar modelo
        self.model = self.create_model(X_train.shape[1], len(self.genre_labels))
        
        print("🎯 Comenzando entrenamiento...")
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1
        )
        
        # Evaluar modelo
        test_loss, test_acc = self.model.evaluate(X_test, y_test, verbose=0)
        print(f"🎉 Precisión final: {test_acc:.4f}")
        
        return history
    
    def save_model(self, model_dir='models'):
        """Guardar modelo entrenado"""
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Guardar modelo Keras
        model_path = os.path.join(model_dir, f'music_model_{timestamp}.h5')
        self.model.save(model_path)
        
        # Guardar scaler
        scaler_path = os.path.join(model_dir, f'scaler_{timestamp}.pkl')
        joblib.dump(self.scaler, scaler_path)
        
        # Guardar metadata
        metadata = {
            'genre_labels': self.genre_labels,
            'timestamp': timestamp,
            'input_dim': self.model.input_shape[1],
            'num_classes': len(self.genre_labels)
        }
        
        metadata_path = os.path.join(model_dir, f'metadata_{timestamp}.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"💾 Modelo guardado en: {model_path}")
        print(f"📊 Scaler guardado en: {scaler_path}")
        print(f"📝 Metadata guardada en: {metadata_path}")
        
        return model_path, scaler_path, metadata_path
    
    def predict_audio(self, audio_path):
        """Predecir género de un archivo de audio"""
        if self.model is None:
            raise Exception("❌ Modelo no entrenado. Entrena primero el modelo.")
        
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
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

# Ejemplo de uso
if __name__ == "__main__":
    trainer = MusicModelTrainer()
    
    # Entrenar con datos (reemplaza con tu directorio de datos)
    try:
        trainer.train_from_directory('audio_dataset')  # Directorio con subcarpetas por género
        
        # Guardar modelo
        model_path, scaler_path, metadata_path = trainer.save_model()
        
        print("\n" + "="*50)
        print("🎵 ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
        print("="*50)
        
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        print("\n💡 CONSEJO: Crea una estructura de directorios como:")
        print("audio_dataset/")
        print("├── rock/")
        print("├── jazz/")
        print("├── classical/")
        print("└── ...") 
