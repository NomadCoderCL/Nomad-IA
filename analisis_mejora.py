import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Embedding, Dropout
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def entrenar_guardar_modelo(texts, labels, model_path='chatbot_model.h5'):
    # Preprocesamiento de textos
    tokenizer = Tokenizer(num_words=10000)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    word_index = tokenizer.word_index

    # Definir la longitud máxima de las secuencias
    max_sequence_length = max(len(seq) for seq in sequences)
    data = pad_sequences(sequences, maxlen=max_sequence_length)

    # Convertir etiquetas a un array numpy
    labels = np.array(labels)

    # Crear el modelo LSTM
    model = Sequential()
    model.add(Embedding(input_dim=len(word_index) + 1, output_dim=128, input_length=max_sequence_length))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='relu'))

    # Compilar el modelo
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'f1_score'])

    # Entrenar el modelo
    model.fit(data, labels, epochs=20, batch_size=64)

    # Guardar el modelo
    model.save(model_path)
    print(f"Modelo guardado en {model_path}")
