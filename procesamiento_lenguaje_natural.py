import numpy as np
import nltk
import spacy
from nltk.corpus import stopwords
from tensorflow.keras.models import load_model
from transformers import pipeline

nltk.download('punkt')
nltk.download('stopwords')

def inicializar_spacy():
    return spacy.load('en_core_web_lg')

def cargar_modelo_lstm():
    try:
        model = load_model('chatbot_model.h5')
        max_n_steps = model.input_shape[1]
        return model, max_n_steps
    except Exception as e:
        print(f"Error cargando el modelo: {e}")
        raise

def cargar_pipeline_analisis_sentimientos():
    return pipeline('sentiment-analysis')

def limpiar_texto(texto, nlp):
    stop_words = set(stopwords.words('spanish'))
    tokens = nltk.word_tokenize(texto.lower())
    tokens = [token for token in tokens if token.isalnum() and token not in stop_words]
    texto_limpio = ' '.join(tokens)
    return texto_limpio

def vectorizar_texto(texto, nlp, max_n_steps):
    vector = nlp(limpiar_texto(texto, nlp)).vector
    vector = np.pad(vector, (0, max_n_steps - len(vector)), 'constant')
    vector = np.expand_dims(np.expand_dims(vector, axis=0), axis=1)
    return vector

def analizar_sentimiento(texto, sentiment_pipeline):
    resultado = sentiment_pipeline(texto)
    return resultado[0]['label'], resultado[0]['score']
