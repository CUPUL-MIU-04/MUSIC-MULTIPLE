// web/model.js - Cliente para la API de Music Multiple AI
class MusicAIClient {
    constructor(apiKey, baseUrl = 'https://music-multiple.onrender.com') {
        this.apiKey = apiKey;
        this.baseUrl = baseUrl;
    }

    async healthCheck() {
        try {
            const response = await fetch(`${this.baseUrl}/health`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error en health check:', error);
            throw error;
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
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error en predicción:', error);
            throw error;
        }
    }

    async predictAudio(audioFile) {
        try {
            const formData = new FormData();
            formData.append('file', audioFile);

            const response = await fetch(`${this.baseUrl}/predict-audio`, {
                method: 'POST',
                headers: {
                    'api-key': this.apiKey
                },
                body: formData
            });

            if (!response.ok) {
                const errorData = await response.json().catch(() => ({ detail: 'Unknown error' }));
                throw new Error(errorData.detail || `HTTP error! status: ${response.status}`);
            }

            const data = await response.json();
            return data;
        } catch (error) {
            console.error('Error en predicción de audio:', error);
            throw error;
        }
    }

    async getModelInfo() {
        try {
            const response = await fetch(`${this.baseUrl}/model-info`);
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            return await response.json();
        } catch (error) {
            console.error('Error obteniendo info del modelo:', error);
            throw error;
        }
    }

    // Método para extraer características de audio (usado internamente)
    async extractAudioFeatures(audioFile) {
        console.log('Procesando archivo de audio:', audioFile.name);
        
        // En una implementación real, aquí procesarías el audio con Web Audio API
        // Por ahora retornamos características simuladas
        const mockFeatures = Array.from({length: 27}, () => (Math.random() * 2 - 1).toFixed(3));
        return mockFeatures;
    }
}

// Función auxiliar para mostrar notificaciones
function showNotification(message, type = 'info') {
    // Crear elemento de notificación
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        border-radius: 10px;
        color: white;
        font-weight: 600;
        z-index: 10000;
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
        transition: all 0.3s ease;
        max-width: 400px;
    `;

    // Colores según el tipo
    const colors = {
        success: '#10b981',
        error: '#ef4444',
        warning: '#f59e0b',
        info: '#3b82f6'
    };

    notification.style.background = colors[type] || colors.info;
    notification.textContent = message;

    document.body.appendChild(notification);

    // Animación de entrada
    setTimeout(() => {
        notification.style.transform = 'translateX(0)';
        notification.style.opacity = '1';
    }, 100);

    // Remover después de 5 segundos
    setTimeout(() => {
        notification.style.transform = 'translateX(100%)';
        notification.style.opacity = '0';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, 5000);
}

// Exportar para usar en otros archivos
if (typeof module !== 'undefined' && module.exports) {
    module.exports = { MusicAIClient, showNotification };
} else {
    window.MusicAIClient = MusicAIClient;
    window.showNotification = showNotification;
        }
