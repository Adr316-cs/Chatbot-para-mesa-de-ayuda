import nltk
from nltk.stem import WordNetLemmatizer
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from keras.optimizers.schedules import ExponentialDecay
import random

# cargar datos de intents
data_file = open('intents_spanish.json', 'r', encoding='utf-8').read()
intents = json.loads(data_file)

lematizer = WordNetLemmatizer()

# Crear listas para palabras, clases y documentos
words = []
classes = []
documents = []
ignore_letters = ['?', '!', '.', ',']

# Procesar cada patrón en los intents
for intent in intents['intents']:
    for pattern in intent['patterns']:
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w, intent['tag']))
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# Lematizar y limpiar palabras
words = [lematizer.lemmatize(w.lower()) for w in words if w not in ignore_letters]
words = sorted(list(set(words)))

classes = sorted(list(set(classes)))

# Guardar palabras y clases en archivos pickle
pickle.dump(words, open('words.pkl', 'wb'))
pickle.dump(classes, open('classes.pkl', 'wb'))

training = []
output_empty = [0] * len(classes)

# Procesar cada documento
for document in documents:
    bag = []
    pattern_words = document[0]
    pattern_words = [lematizer.lemmatize(word.lower()) for word in pattern_words]
    for w in words:
        bag.append(1) if w in pattern_words else bag.append(0)
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

# Mezclar los datos de entrenamiento
random.shuffle(training)

# Separar las características y etiquetas
train_x = [row[0] for row in training]
train_y = [row[1] for row in training]

# Convertir a arrays de numpy
train_x = np.array(train_x)
train_y = np.array(train_y)
 
# Crear el modelo de red neuronal
model = Sequential()
model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(train_y[0]), activation='softmax'))

# Definir el optimizador con decaimiento exponencial de la tasa de aprendizaje
lr_schedule = ExponentialDecay(
     initial_learning_rate=0.01,
        decay_steps=1000,
        decay_rate=0.96)

# Compilar el modelo
sgd = SGD(learning_rate=lr_schedule, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

# Entrenar el modelo
hist = model.fit(np.array(train_x), np.array(train_y), epochs=200, batch_size=5, verbose=1)

# Guardar el modelo entrenado
model.save('chatbot_model.h5', hist)
print("Modelo entrenado y guardado como 'chatbot_model.h5'")