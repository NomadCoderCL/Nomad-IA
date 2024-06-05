
import tkinter as tk
from interfaz_usuario import ChatbotGUI
from integraciones import inicializar_spacy, inicializar_tts
from procesamiento_lenguaje_natural import cargar_modelo_lstm, cargar_pipeline_analisis_sentimientos
from base_conocimiento import crear_base_datos

# Inicializar las integraciones
nlp = inicializar_spacy()
sentiment_pipeline = cargar_pipeline_analisis_sentimientos()
tts_engine = inicializar_tts()

# Cargar el modelo de LSTM
lstm_model, max_n_steps = cargar_modelo_lstm()

# Crear la base de datos
crear_base_datos()

# Iniciar la interfaz gráfica
root = tk.Tk()
app = ChatbotGUI(root, nlp, lstm_model, max_n_steps, sentiment_pipeline, tts_engine)
root.mainloop()
