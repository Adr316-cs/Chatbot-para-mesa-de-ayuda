# chatbot.py
"""Carga el modelo entrenado y resuelve la respuesta a un mensaje del usuario."""
import json
import os
import pickle
import random
import re

import numpy as np
from keras.models import load_model

from preprocess import (
    CLASSES_PATH,
    INTENTS_PATH,
    MODEL_PATH,
    WORDS_PATH,
    bag_of_words,
    ensure_nltk_data,
    strip_accents,
    tiene_palabras_de_contenido,
)
from registro import UMBRAL_DUDOSO, registrar
from tickets import crear_ticket

# Confianza minima para dar por buena una intencion. Por debajo de esto se
# responde con el fallback en lugar de arriesgar una respuesta equivocada.
# Medido sobre el dataset actual: las parafrasis legitimas quedan por encima de
# 0.62 y las preguntas fuera de tema por debajo de 0.38, asi que 0.50 separa
# bien ambos casos. Si se amplian los patrones conviene volver a medirlo.
ERROR_THRESHOLD = 0.50

ensure_nltk_data()

with open(INTENTS_PATH, "r", encoding="utf-8") as f:
    intents = json.load(f)

_faltantes = [p for p in (WORDS_PATH, CLASSES_PATH, MODEL_PATH) if not os.path.exists(p)]
if _faltantes:
    raise FileNotFoundError(
        "Faltan los archivos del modelo entrenado:\n  "
        + "\n  ".join(_faltantes)
        + "\n\nEjecuta primero:  python training_chatbot.py"
    )

with open(WORDS_PATH, "rb") as f:
    words = pickle.load(f)
with open(CLASSES_PATH, "rb") as f:
    classes = pickle.load(f)
model = load_model(MODEL_PATH)

# Respuestas de fallback tomadas del JSON. Esta intencion no es una clase del
# modelo (no tiene patrones): se usa cuando ninguna clase supera el umbral.
FALLBACK_RESPONSES = next(
    (i["responses"] for i in intents["intents"] if i["tag"] == "fallback"),
    ["Lo siento, no entendi tu mensaje. ¿Puedes reformularlo?"],
)

# ---------------------------------------------------------------------------
# Menu numerado
#
# La navegacion NO pasa por el modelo. Elegir una opcion de una lista es una
# instruccion exacta, no algo que haya que adivinar: si se deja al
# clasificador, un '1' o un 'menu' terminan en la intencion que mas se le
# parezca (con este dataset, 'menu' caia en 'despedida').
# ---------------------------------------------------------------------------
MENU = intents.get("menu", [])

# Etiqueta del boton para regresar al menu.
VOLVER_AL_MENU = "Ver el menú"

# Se comparan ya normalizadas (sin acentos y en minusculas).
PALABRAS_MENU = {"menu", "opciones", "inicio", "ayuda", "volver", "regresar",
                 "tramites", "ver el menu"}


def _normalizar(texto):
    return strip_accents(texto.strip().lower())


def texto_menu():
    return ("Esto es lo que puedo hacer por ti. Elige una opción, o escríbeme "
            "tu pregunta directamente:")


def _respuesta_de_tag(tag):
    for intent in intents["intents"]:
        if intent["tag"] == tag:
            respuestas = [r for r in intent["responses"] if r.strip()]
            if respuestas:
                return random.choice(respuestas)
    return None


def _menu_completo():
    # 'Reportar un problema' se agrega al final: no es un tramite, es la salida
    # para lo que el bot no resuelve.
    return {"response": texto_menu(),
            "options": [o["label"] for o in MENU] + [ETIQUETA_TICKET]}


def _respuesta_de_opcion(opcion):
    respuesta = _respuesta_de_tag(opcion["tag"])
    if not respuesta:
        return None
    return {"response": f"{opcion['label']}\n\n{respuesta}",
            "options": [VOLVER_AL_MENU]}


def responder_menu(mensaje):
    """Atiende el menu y la seleccion de una opcion.

    Devuelve {'response': str, 'options': [str]} o None si el mensaje no es una
    interaccion de menu, para que siga su camino hacia el clasificador.
    """
    texto = _normalizar(mensaje)

    if texto in PALABRAS_MENU:
        return _menu_completo()

    # Asi llegan los clics de los botones: el frontend manda la etiqueta tal
    # cual, sin numeros de por medio.
    for opcion in MENU:
        if _normalizar(opcion["label"]) == texto:
            respuesta = _respuesta_de_opcion(opcion)
            if respuesta:
                return respuesta

    # Atajo por numero, por si alguien lo escribe. Solo si el mensaje es
    # unicamente un numero: 'paso 1' o 'articulo 3' son preguntas de verdad y
    # deben ir al modelo.
    if re.fullmatch(r"\d+", texto):
        indice = int(texto)
        if 1 <= indice <= len(MENU):
            respuesta = _respuesta_de_opcion(MENU[indice - 1])
            if respuesta:
                return respuesta
        respuesta = _menu_completo()
        respuesta["response"] = (f"No tengo una opción con el número {indice}.\n\n"
                                 + respuesta["response"])
        return respuesta

    return None


# ---------------------------------------------------------------------------
# Escalamiento a ticket
#
# El menu resuelve el "como hago X". Esto atiende la otra mitad: reportar un
# problema que el bot no puede resolver y pasarlo a una persona.
#
# Es un formulario de varios turnos, asi que necesita memoria: el estado de la
# conversacion lo guarda app.py en la sesion y lo pasa en cada llamada.
# ---------------------------------------------------------------------------
CAMPOS_TICKET = intents.get("ticket", {}).get("campos", [])

ETIQUETA_TICKET = "Reportar un problema"
ETIQUETA_CANCELAR = "Cancelar"
ETIQUETA_OMITIR = "Omitir"

# Peticiones EXPLICITAS de escalamiento: el usuario ya decidio que quiere un
# agente, no hay que intentar resolverlo antes.
PALABRAS_TICKET = {
    "reportar un problema", "reportar problema", "reportar", "ticket",
    "levantar un ticket", "levantar ticket", "abrir un ticket", "crear ticket",
    "quiero un ticket", "necesito un ticket",
    "hablar con un agente", "hablar con una persona", "hablar con alguien",
    "necesito ayuda de un agente", "atencion humana",
}

# Frases vagas de problema. NO abren el formulario: primero se pide el detalle
# para intentar resolverlo, y solo si el bot no puede se ofrece escalar.
# Antes "tengo un problema" saltaba directo al ticket sin intentar ayudar.
PALABRAS_PROBLEMA_VAGO = {
    "tengo un problema", "tengo un error", "hay un problema", "problema",
    "tengo un inconveniente", "necesito ayuda", "ayudame", "auxilio",
    "no me funciona", "no funciona", "algo salio mal", "tengo una falla",
}
PALABRAS_CANCELAR = {"cancelar", "cancela", "salir", "olvidalo", "ya no",
                     "mejor no", "detener"}
PALABRAS_OMITIR = {"omitir", "no", "ninguno", "no tengo", "no aplica",
                   "n/a", "na", "-", "sin dato"}

_CORREO = re.compile(r"^[^@\s]+@[^@\s]+\.[a-z]{2,}$", re.IGNORECASE)


def _pregunta_de(indice):
    """Texto de la pregunta con su avance y los botones que corresponden."""
    campo = CAMPOS_TICKET[indice]
    avance = f"Pregunta {indice + 1} de {len(CAMPOS_TICKET)}"
    opciones = [ETIQUETA_CANCELAR]
    if not campo.get("requerido", True):
        opciones.insert(0, ETIQUETA_OMITIR)
    return {"response": f"{avance}\n\n{campo['pregunta']}", "options": opciones}


def _iniciar_ticket(estado):
    if not CAMPOS_TICKET:
        return {"response": "El registro de tickets no está configurado.",
                "options": [VOLVER_AL_MENU]}, estado

    estado["ticket"] = {"paso": 0, "datos": {}}
    estado["fallos"] = 0
    primera = _pregunta_de(0)
    return {
        "response": "Con gusto levanto un reporte para que un agente te atienda.\n\n"
                    "Te haré unas preguntas breves. Puedes escribir «cancelar» "
                    "en cualquier momento.\n\n" + primera["response"],
        "options": primera["options"],
    }, estado


def _paso_ticket(mensaje, estado):
    """Procesa una respuesta del usuario dentro del formulario."""
    texto = _normalizar(mensaje)
    flujo = estado["ticket"]

    if texto in PALABRAS_CANCELAR:
        estado.pop("ticket", None)
        return {"response": "Listo, cancelé el reporte. ¿En qué más te puedo ayudar?",
                "options": [VOLVER_AL_MENU, ETIQUETA_TICKET]}, estado

    campo = CAMPOS_TICKET[flujo["paso"]]
    requerido = campo.get("requerido", True)
    valor = mensaje.strip()

    if not requerido and texto in PALABRAS_OMITIR:
        valor = ""
    elif not valor or (requerido and texto in PALABRAS_OMITIR):
        pregunta = _pregunta_de(flujo["paso"])
        return {"response": "Necesito este dato para poder registrar el reporte.\n\n"
                            + pregunta["response"],
                "options": pregunta["options"]}, estado
    elif campo.get("tipo") == "correo" and not _CORREO.match(valor):
        pregunta = _pregunta_de(flujo["paso"])
        return {"response": "Ese correo no parece válido. Revísalo, por favor "
                            "(ejemplo: nombre@dominio.gob.mx).\n\n" + pregunta["response"],
                "options": pregunta["options"]}, estado

    flujo["datos"][campo["id"]] = valor
    flujo["paso"] += 1

    if flujo["paso"] < len(CAMPOS_TICKET):
        return _pregunta_de(flujo["paso"]), estado

    # Formulario completo: se registra y se cierra el flujo.
    datos = flujo["datos"]
    estado.pop("ticket", None)

    try:
        folio, destinos = crear_ticket(datos)
    except Exception as exc:
        print(f"[tickets] Error al registrar: {exc}")
        return {"response": "Tuve un problema al registrar el ticket. "
                            "Vuelve a intentarlo en un momento, por favor.",
                "options": [ETIQUETA_TICKET, VOLVER_AL_MENU]}, estado

    aviso = ("" if "sheets" in destinos else
             "\n\n(El reporte quedó guardado localmente; el área de sistemas lo "
             "sincronizará con la hoja de tickets.)")

    return {
        "response": f"Listo, tu reporte quedó registrado.\n\n"
                    f"Folio: {folio}\n"
                    f"Estado: Nuevo\n\n"
                    f"Guarda ese folio para dar seguimiento. Un agente se pondrá "
                    f"en contacto al correo {datos.get('correo', '')}.{aviso}",
        "options": [VOLVER_AL_MENU],
    }, estado


def _clasificar(sentence):
    """Devuelve (candidatas_sobre_umbral, mejor_candidata, su_confianza).

    La mejor candidata se devuelve aunque no supere el umbral: la bitacora
    necesita saber que estuvo a punto de contestar, y asi el modelo se ejecuta
    una sola vez por mensaje.
    """
    bow = bag_of_words(sentence, words)

    # Dos casos sin senal real de entrada, que van directo al fallback:
    #
    # 1. Ninguna palabra del vocabulario: el vector va en ceros y la red
    #    devuelve su prediccion por defecto con mucha confianza (con este
    #    dataset, 'despedida' al 0.88).
    # 2. Solo coinciden palabras vacias ('quien', 'el', 'de'...), que estan en
    #    casi todos los patrones y no distinguen nada.
    if not bow.any() or not tiene_palabras_de_contenido(sentence, words):
        return [], None, 0.0

    res = model.predict(np.array([bow]), verbose=0)[0]
    orden = sorted(enumerate(res), key=lambda x: x[1], reverse=True)
    mejor_i, mejor_p = orden[0]
    sobre_umbral = [{"intent": classes[i], "probability": str(p)}
                    for i, p in orden if p > ERROR_THRESHOLD]
    return sobre_umbral, classes[mejor_i], float(mejor_p)


def predict_class(sentence):
    """Intenciones candidatas que superan el umbral, ordenadas por probabilidad.

    Puede devolver una lista vacia; quien la consuma debe contemplarlo.
    """
    return _clasificar(sentence)[0]


def get_response(intents_list, intents_json):
    """Elige la respuesta para la intencion mas probable.

    Si la lista viene vacia (nada supero el umbral) responde con el fallback.
    Antes esto lanzaba IndexError y el usuario se quedaba sin respuesta.
    """
    if not intents_list:
        return random.choice(FALLBACK_RESPONSES)

    tag = intents_list[0]["intent"]
    for intent in intents_json["intents"]:
        if intent["tag"] == tag:
            respuestas = [r for r in intent["responses"] if r.strip()]
            if not respuestas:
                return random.choice(FALLBACK_RESPONSES)
            return random.choice(respuestas)

    return random.choice(FALLBACK_RESPONSES)


def responder(mensaje, estado=None):
    """Punto de entrada unico que usa la aplicacion.

    Devuelve (resultado, estado). 'resultado' es {'response', 'options'};
    'estado' es la memoria de la conversacion, que app.py guarda en la sesion
    y devuelve en la siguiente llamada.

    Orden: formulario de ticket en curso -> peticion de ticket -> menu ->
    clasificador. Lo determinista va antes que el modelo.
    """
    estado = dict(estado or {})
    texto = _normalizar(mensaje)

    # Con un formulario abierto, todo lo que llegue es respuesta a la pregunta
    # en turno. Si no, un "Juan Pérez" acabaria en el clasificador.
    if estado.get("ticket"):
        return _paso_ticket(mensaje, estado)

    if texto in PALABRAS_TICKET:
        return _iniciar_ticket(estado)

    # "Tengo un problema" no dice cual. Se pide el detalle para intentar
    # resolverlo primero; el ticket es el ultimo recurso, no el primero.
    if texto in PALABRAS_PROBLEMA_VAGO:
        estado["fallos"] = 0
        return {
            "response": "Claro, cuéntame qué está pasando y trato de resolverlo.\n\n"
                        "Descríbeme el problema: en qué trámite estás, qué "
                        "intentabas hacer y qué te muestra el sistema.\n\n"
                        "Si no logro ayudarte, levantamos un reporte para que un "
                        "agente lo atienda.",
            "options": [VOLVER_AL_MENU],
        }, estado

    respuesta_menu = responder_menu(mensaje)
    if respuesta_menu is not None:
        estado["fallos"] = 0
        return respuesta_menu, estado

    # Solo se llega aqui con una pregunta de verdad: lo que se escribe dentro
    # del formulario de ticket (nombre, correo, telefono) sale antes y nunca
    # toca la bitacora.
    intents_list, candidato, confianza = _clasificar(mensaje)
    respuesta = get_response(intents_list, intents)

    if intents_list:
        estado["fallos"] = 0
        # Contesto, pero sin estar seguro: es donde mas probable es que haya
        # respondido otra cosa, asi que queda anotado para revisarlo.
        if confianza < UMBRAL_DUDOSO:
            registrar(mensaje, "confianza_baja", candidato, confianza)
        # Tras responder un problema se ofrece escalar, por si la respuesta no
        # resolvio el caso. En una pregunta de tramite no hace falta.
        es_problema = intents_list[0]["intent"].startswith("problema_")
        opciones = [ETIQUETA_TICKET, VOLVER_AL_MENU] if es_problema else []
        return {"response": respuesta, "options": opciones}, estado

    registrar(mensaje, "sin_coincidencia", candidato, confianza)

    # No se entendio. Se ofrece salida en vez de dejar al usuario atorado, y a
    # la segunda seguida se propone abiertamente escalar a un agente.
    estado["fallos"] = estado.get("fallos", 0) + 1
    if estado["fallos"] >= 2:
        return {
            "response": "Parece que esto se sale de lo que puedo resolver.\n\n"
                        "¿Quieres que levante un reporte para que un agente te "
                        "atienda? También puedo mostrarte la lista de trámites.",
            "options": [ETIQUETA_TICKET, VOLVER_AL_MENU],
        }, estado

    return {"response": respuesta,
            "options": [ETIQUETA_TICKET, VOLVER_AL_MENU]}, estado
