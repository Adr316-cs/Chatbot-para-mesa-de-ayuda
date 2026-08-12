# registro.py
"""Bitacora de preguntas que el bot no supo contestar.

Es el insumo mas util para mejorar el chatbot: dice que esta preguntando la
gente de verdad, en sus palabras, en vez de adivinar que patrones agregar.

Se registran dos casos:

  sin_coincidencia : ninguna intencion supero el umbral -> respondio el fallback.
  confianza_baja   : contesto, pero con poca confianza; es donde el bot tiene
                     mas probabilidad de haber respondido otra cosa.

NO se registra nada escrito dentro del formulario de ticket: ahi el usuario
teclea su nombre, correo y telefono. Esa separacion vive en chatbot.py, que
solo llama a este modulo desde la rama del clasificador.
"""
import csv
import datetime
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "no_entendidas.csv")

COLUMNAS = ["fecha", "motivo", "mensaje", "mejor_candidato", "confianza"]

# Debajo de esta confianza, una respuesta se considera dudosa y se anota.
# El umbral para responder es 0.50 (ERROR_THRESHOLD en chatbot.py), asi que
# esto marca la franja 0.50-0.70: contesto, pero sin estar seguro.
UMBRAL_DUDOSO = 0.70

# Longitud maxima que se guarda del mensaje, por si alguien pega un texto enorme.
MAX_LARGO = 300

_candado = threading.Lock()


def registrar(mensaje, motivo, candidato=None, confianza=0.0):
    """Anota una pregunta en la bitacora.

    Nunca interrumpe la conversacion: si falla la escritura, se avisa por
    consola y el usuario recibe su respuesta igual.
    """
    try:
        fila = [
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            motivo,
            (mensaje or "").strip()[:MAX_LARGO],
            candidato or "",
            f"{confianza:.2f}",
        ]
        with _candado:
            nuevo = not os.path.exists(CSV_PATH)
            with open(CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
                escritor = csv.writer(f)
                if nuevo:
                    escritor.writerow(COLUMNAS)
                escritor.writerow(fila)
    except Exception as exc:
        print(f"[registro] No se pudo anotar la pregunta: {exc}")
