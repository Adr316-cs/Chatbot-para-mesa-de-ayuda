# preprocess.py
"""Normalizacion de texto y rutas compartidas por el entrenamiento y la inferencia.

IMPORTANTE: tanto training_chatbot.py como chatbot.py deben limpiar el texto
con estas mismas funciones. Si el texto se procesa distinto al entrenar y al
predecir, el bag-of-words no coincide y el modelo falla aunque este bien
entrenado.
"""
import os
import unicodedata

import nltk
import numpy as np
from nltk.stem import SnowballStemmer

# Todas las rutas se resuelven contra la carpeta de este archivo, no contra el
# directorio desde el que se ejecuta el script. Asi funciona igual si corres
# `python app.py` dentro de chatbot/ o `python chatbot/app.py` desde la raiz.
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

INTENTS_PATH = os.path.join(BASE_DIR, "intents_spanish.json")
WORDS_PATH = os.path.join(BASE_DIR, "words.pkl")
CLASSES_PATH = os.path.join(BASE_DIR, "classes.pkl")
MODEL_PATH = os.path.join(BASE_DIR, "chatbot_model.keras")

# Signos que se descartan del vocabulario. Se incluyen los de apertura del
# espanol ('¿', '¡'), que el codigo anterior dejaba entrar como palabras.
IGNORE_TOKENS = {"?", "¿", "!", "¡", ".", ",", ";", ":", '"', "'", "(", ")", "-", "_"}

# Stemmer en espanol. El WordNetLemmatizer que se usaba antes es de ingles y
# sobre texto en espanol practicamente no hace nada.
_stemmer = SnowballStemmer("spanish")


def ensure_nltk_data():
    """Descarga los recursos de NLTK la primera vez. Evita el LookupError
    tipico al montar el proyecto en una maquina nueva."""
    for resource, path in (
        ("punkt", "tokenizers/punkt"),
        ("punkt_tab", "tokenizers/punkt_tab"),
        ("stopwords", "corpora/stopwords"),
    ):
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(resource, quiet=True)


def strip_accents(text):
    """'informacion' == 'información'.

    Los patrones del JSON estan escritos sin acentos pero los usuarios
    escriben con acentos. Sin esta normalizacion son tokens distintos y
    nunca coinciden.
    """
    descompuesto = unicodedata.normalize("NFKD", text)
    return "".join(c for c in descompuesto if not unicodedata.combining(c))


def clean_up_sentence(sentence):
    """Convierte una frase en la lista de raices (stems) que usa el modelo."""
    sentence = strip_accents(sentence.lower())
    tokens = nltk.word_tokenize(sentence, language="spanish")
    return [
        _stemmer.stem(t)
        for t in tokens
        if t not in IGNORE_TOKENS and any(c.isalnum() for c in t)
    ]


def bag_of_words(sentence, words):
    """Vector binario de presencia de cada palabra del vocabulario."""
    sentence_words = set(clean_up_sentence(sentence))
    return np.array([1 if w in sentence_words else 0 for w in words])


_vacias = None


def palabras_vacias():
    """Raices de las palabras vacias del espanol ('el', 'de', 'que', 'quien'...).

    Se calculan una sola vez y ya stemmizadas, para poder compararlas contra el
    vocabulario, que tambien esta stemmizado.
    """
    global _vacias
    if _vacias is None:
        from nltk.corpus import stopwords

        _vacias = {_stemmer.stem(strip_accents(p.lower()))
                   for p in stopwords.words("spanish")}
    return _vacias


def tiene_palabras_de_contenido(sentence, words):
    """True si el mensaje coincide con el vocabulario en alguna palabra con
    significado propio.

    Las palabras vacias aparecen en casi todos los patrones, asi que coincidir
    solo en ellas no aporta senal: 'quien gano el partido' coincide en 'quien'
    y 'el', y con eso la red llegaba a dar una respuesta equivocada con 0.68 de
    confianza.
    """
    comunes = set(clean_up_sentence(sentence)) & set(words)
    return bool(comunes - palabras_vacias())
