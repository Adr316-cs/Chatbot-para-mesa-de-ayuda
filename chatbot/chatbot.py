import random
import json
import pickle
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from keras.models import load_model
import gspread
from oauth2client.service_account import ServiceAccountCredentials

# Inicializar lematizador y cargar datos
lematizer = WordNetLemmatizer()
intents = json.loads(open('intents_spanish.json', 'r', encoding='utf-8').read())

# Cargar palabras, clases y modelo entrenado
words = pickle.load(open('words.pkl', 'rb'))
classes = pickle.load(open('classes.pkl', 'rb'))
model = load_model('chatbot_model.h5')

# Función para crear tickets en Google Sheets
def crear_ticket_google(asunto, descripcion):
    scope = [
        "https://spreadsheets.google.com/feeds",
        "https://www.googleapis.com/auth/spreadsheets",
        "https://www.googleapis.com/auth/drive.file",
        "https://www.googleapis.com/auth/drive"
    ]

    creds = ServiceAccountCredentials.from_json_keyfile_name(r"C:\Users\luisa\Documents\PROYECTO ESTADIAS\chatbot\credenciales.json", scope)
    client = gspread.authorize(creds)

    # Abrir la hoja (asegúrate que se llame exactamente así en Google Sheets)
    sheet = client.open("tickets").sheet1

    # Generar ID automático
    nuevo_id = len(sheet.get_all_values())

    # Agregar fila
    sheet.append_row([nuevo_id, asunto, descripcion])

    return nuevo_id

# Funciones de procesamiento de texto
def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lematizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]

def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if i['tag'] == tag:
            result = random.choice(i['responses'])
            if tag == "error_sistema":
                ticket_id = crear_ticket_google(
                    "Error reportado por chatbot",
                    "El usuario reportó un error del sistema."
                )
                result += f" Ticket guardado con ID {ticket_id}."
            return result
    return "Lo siento, no entendí tu mensaje."