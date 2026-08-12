# training_chatbot.py
"""Entrena el clasificador de intenciones.

Uso:    python training_chatbot.py
Genera: words.pkl, classes.pkl, chatbot_model.keras

Antes de entrenar valida el JSON de intenciones y avisa de los problemas que
degradan la calidad del modelo (patrones duplicados con etiquetas distintas,
etiquetas repetidas, intenciones sin patrones).
"""
import json
import pickle
import random
from collections import Counter, defaultdict

import numpy as np
from keras.layers import Dense, Dropout, Input
from keras.models import Sequential
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay

from preprocess import (
    CLASSES_PATH,
    INTENTS_PATH,
    MODEL_PATH,
    WORDS_PATH,
    clean_up_sentence,
    ensure_nltk_data,
)

ensure_nltk_data()

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

# ---------------------------------------------------------------------------
# Validacion del dataset
# ---------------------------------------------------------------------------
problemas = 0
etiquetas_repetidas = [t for t, c in Counter(i["tag"] for i in intents["intents"]).items() if c > 1]
for tag in etiquetas_repetidas:
    print(f"  AVISO: la etiqueta '{tag}' esta definida mas de una vez.")
    problemas += 1

# Cada opcion del menu debe apuntar a una intencion que exista y que tenga
# respuesta; si no, el usuario elige un numero y no recibe nada.
tags_existentes = {i["tag"] for i in intents["intents"]}
for opcion in intents.get("menu", []):
    if opcion["tag"] not in tags_existentes:
        print(f"  AVISO: la opcion de menu '{opcion['label']}' apunta a la etiqueta "
              f"'{opcion['tag']}', que no existe.")
        problemas += 1

# ---------------------------------------------------------------------------
# Construccion del vocabulario y los documentos de entrenamiento
# ---------------------------------------------------------------------------
words = []
classes = []
documents = []
patrones_vistos = defaultdict(set)

for intent in intents["intents"]:
    tag = intent["tag"]
    utilizables = [p for p in intent["patterns"] if p.strip()]

    # 'fallback' no puede ser una clase del modelo: no tiene patrones, asi que
    # nunca podria predecirse. Su respuesta la entrega chatbot.py cuando
    # ninguna clase supera el umbral de confianza.
    if tag == "fallback":
        continue
    if not utilizables:
        print(f"  AVISO: la intencion '{tag}' no tiene patrones utiles; se omite.")
        problemas += 1
        continue

    if tag not in classes:
        classes.append(tag)

    for pattern in utilizables:
        tokens = clean_up_sentence(pattern)
        if not tokens:
            continue
        clave = " ".join(sorted(tokens))
        patrones_vistos[clave].add(tag)
        words.extend(tokens)
        documents.append((tokens, tag))

for clave, tags in patrones_vistos.items():
    if len(tags) > 1:
        print(f"  AVISO: un mismo patron esta etiquetado como {sorted(tags)}. "
              f"El modelo no puede aprender ambos.")
        problemas += 1

words = sorted(set(words))
classes = sorted(set(classes))

print(f"\n{len(documents)} patrones | {len(classes)} intenciones | {len(words)} palabras en vocabulario")
por_intencion = Counter(tag for _, tag in documents)
escasas = [t for t, n in por_intencion.items() if n < 3]
if escasas:
    print(f"  AVISO: {len(escasas)} de {len(classes)} intenciones tienen menos de 3 patrones. "
          f"El modelo memorizara en vez de generalizar; lo ideal son 8-15 por intencion.")
if problemas:
    print(f"\n  Se detectaron {problemas} problema(s) en el dataset (ver arriba).\n")

with open(WORDS_PATH, "wb") as f:
    pickle.dump(words, f)
with open(CLASSES_PATH, "wb") as f:
    pickle.dump(classes, f)

# ---------------------------------------------------------------------------
# Vectorizacion
# ---------------------------------------------------------------------------
training = []
output_empty = [0] * len(classes)

for tokens, tag in documents:
    presentes = set(tokens)
    bag = [1 if w in presentes else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(tag)] = 1
    training.append([bag, output_row])

random.shuffle(training)
train_x = np.array([row[0] for row in training])
train_y = np.array([row[1] for row in training])

# ---------------------------------------------------------------------------
# Modelo
# ---------------------------------------------------------------------------
model = Sequential()
model.add(Input(shape=(train_x.shape[1],)))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(classes), activation="softmax"))

lr_schedule = ExponentialDecay(
    initial_learning_rate=0.01,
    decay_steps=1000,
    decay_rate=0.96,
)
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

model.fit(train_x, train_y, epochs=200, batch_size=5, verbose=1)

model.save(MODEL_PATH)

# Comprobacion: el modelo deberia acertar sus propios patrones de entrenamiento.
# No mide generalizacion, pero detecta si algo se rompio en el pipeline.
preds = model.predict(train_x, verbose=0)
aciertos = float(np.mean(np.argmax(preds, axis=1) == np.argmax(train_y, axis=1)))
print(f"\nModelo guardado en {MODEL_PATH}")
print(f"Acierto sobre los patrones de entrenamiento: {aciertos:.1%}")
