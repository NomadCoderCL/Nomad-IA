import tkinter as tk
from tkinter import scrolledtext
from base_conocimiento import guardar_conversacion
from procesamiento_lenguaje_natural import vectorizar_texto, analizar_sentimiento

class ChatbotGUI:
    def __init__(self, master, nlp, model, max_n_steps, sentiment_pipeline, tts_engine):
        self.master = master
        self.nlp = nlp
        self.model = model
        self.max_n_steps = max_n_steps
        self.sentiment_pipeline = sentiment_pipeline
        self.tts_engine = tts_engine

        self.master.title("Chatbot")
        self.master.geometry("500x400")
        self.bg_color = "#f0f0f0"
        self.text_color = "#333333"
        self.button_color = "#4CAF50"
        self.entry_bg_color = "#ffffff"
        self.entry_fg_color = "#333333"

        self.master.configure(bg=self.bg_color)
        self.chat_history = scrolledtext.ScrolledText(self.master, state=tk.DISABLED, bg=self.entry_bg_color, fg=self.text_color, font=("Arial", 12))
        self.chat_history.grid(row=0, column=0, columnspan=3, padx=10, pady=10)

        self.entry_field = tk.Entry(self.master, bg=self.entry_bg_color, fg=self.entry_fg_color, font=("Arial", 12))
        self.entry_field.grid(row=1, column=0, padx=10, pady=10, sticky="ew")

        self.send_button = tk.Button(self.master, text="Enviar", command=self.enviar_mensaje, bg=self.button_color, fg=self.entry_fg_color)
        self.send_button.grid(row=1, column=1, padx=10, pady=10)

        self.close_button = tk.Button(self.master, text="Cerrar", command=self.master.destroy, bg=self.button_color, fg=self.entry_fg_color)
        self.close_button.grid(row=1, column=2, padx=10, pady=10)

        self.master.grid_rowconfigure(0, weight=1)
        self.master.grid_columnconfigure(0, weight=1)

        self.responses = {
            0: "Hola",
            1: "Estoy bien, gracias",
            2: "Son las 3 PM",
            3: "Adiós",
            4: "Sí, ¿en qué puedo ayudarte?",
            5: "Estoy analizando...",
            6: "Hoy es el 18 de mayo de 2024",
            7: "De nada",
            8: "Lo siento",
            9: "Claro, ¿puedes repetirlo por favor?",
            10: "Buenos días, tardes o noches",
            11: "Está soleado",
            12: "Soy tu asistente virtual",
            13: "Igualmente, un placer conocerte"
        }

    def enviar_mensaje(self):
        mensaje_usuario = self.entry_field.get()
        if mensaje_usuario:
            self.chat_history.config(state=tk.NORMAL)
            self.chat_history.insert(tk.END, "Tú: " + mensaje_usuario + '\n\n')
            self.chat_history.config(state=tk.DISABLED)

            sentimiento, confianza = analizar_sentimiento(mensaje_usuario, self.sentiment_pipeline)
            print(f"Sentimiento: {sentimiento}, Confianza: {confianza}")

            etiqueta_predicha = self.predict_intent(mensaje_usuario)
            respuesta_chatbot = self.responses.get(etiqueta_predicha, "No entiendo la pregunta.")
            self.chat_history.config(state=tk.NORMAL)
            self.chat_history.insert(tk.END, "Chatbot: " + respuesta_chatbot + "\n\n")
            self.chat_history.config(state=tk.DISABLED)

            self.speak(respuesta_chatbot)
            guardar_conversacion(mensaje_usuario, respuesta_chatbot)
            self.entry_field.delete(0, tk.END)

    def predict_intent(self, text):
        vector = vectorizar_texto(text, self.nlp, self.max_n_steps)
        prediccion = self.model.predict(vector)
        etiqueta_predicha = np.argmax(prediccion, axis=1)[0]
        return etiqueta_predicha

    def speak(self, text):
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

