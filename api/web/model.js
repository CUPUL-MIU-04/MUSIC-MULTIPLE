// web/model.js - Cliente para la API de Music AI

class MusicAIClient {
    constructor(apiKey, baseUrl = 'https://tu-api-desplegada.com') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            return await response.json();
        } catch (error) {
            console.error('Error en health check:', error);
            return { status: 'unreachable', error: error.message };
        }
    }

    async predict(audioFeatures) {
        try {
            const response = await fetch(`${this.baseUrl}/predict`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                    'api-key': this.apiKey
                },
                body: JSON.stringify({
                    features: audioFeatures
                })
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error en predicción:', error);
            throw error;
        }
    }

    // Método para procesar audio (ejemplo)
    async processAudioFile(audioFile) {
        // Aquí puedes agregar lógica para extraer features del audio
        const features = await this.extractAudioFeatures(audioFile);
        return this.predict(features);
    }

    async extractAudioFeatures(audioFile) {
        // Ejemplo simulado - reemplaza con tu lógica real de extracción de features
        console.log('Procesando archivo de audio:', audioFile.name);
        
        // Simulación: retorna features aleatorios
        const mockFeatures = Array.from({length: 13}, () => Math.random());
        return mockFeatures;
    }
}

// Ejemplo de uso:
/*
// Inicializar cliente
const client = new MusicAIClient('music_ai_key_gerardo_2024');

// Verificar salud de la API
client.healthCheck().then(health => {
    console.log('Estado de la API:', health);
});

// Hacer predicción con features de audio
const audioFeatures = [0.1, 0.5, 0.3, 0.8, 0.2, 0.9, 0.4, 0.6, 0.7, 0.1, 0.3, 0.5, 0.8];
client.predict(audioFeatures).then(result => {
    console.log('Predicción:', result);
});
*/

// Exportar para usar en otros archivos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = MusicAIClient;
} else {
    window.MusicAIClient = MusicAIClient;
}
