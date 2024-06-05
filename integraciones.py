import spacy
import pyttsx3
from transformers import pipeline

def inicializar_spacy():
    return spacy.load('en_core_web_lg')

def inicializar_tts():
    return pyttsx3.init()

def cargar_pipeline_analisis_sentimientos():
    return pipeline('sentiment-analysis')

