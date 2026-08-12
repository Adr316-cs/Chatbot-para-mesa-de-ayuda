# evaluar.py
"""Mide que tan bien responde el clasificador a preguntas reales.

Uso:  python evaluar.py [--detalle]

Los casos viven en evaluacion.json y estan redactados como los escribiria un
usuario, no como los patrones de entrenamiento. El acierto que reporta
training_chatbot.py es sobre sus propios patrones (siempre ~100%): mide
memorizacion. Este script mide generalizacion, que es lo que importa.

Ejecutalo antes y despues de tocar intents_spanish.json para saber si un
cambio mejoro o empeoro el bot.
"""
import json
import os
import sys

from preprocess import BASE_DIR

EVAL_PATH = os.path.join(BASE_DIR, "evaluacion.json")


def main():
    detalle = "--detalle" in sys.argv

    with open(EVAL_PATH, "r", encoding="utf-8") as f:
        casos = json.load(f)["casos"]

    from chatbot import predict_class  # se importa aqui: carga el modelo

    aciertos = aciertos_fallback = 0
    total_intencion = total_fallback = 0
    fallos = []

    for caso in casos:
        pregunta, esperado = caso["pregunta"], caso["esperado"]
        resultado = predict_class(pregunta)
        obtenido = resultado[0]["intent"] if resultado else None
        confianza = float(resultado[0]["probability"]) if resultado else 0.0

        if esperado is None:
            total_fallback += 1
            if obtenido is None:
                aciertos_fallback += 1
            else:
                fallos.append((pregunta, "(fallback)", obtenido, confianza))
        else:
            total_intencion += 1
            if obtenido == esperado:
                aciertos += 1
            else:
                fallos.append((pregunta, esperado, obtenido or "(fallback)", confianza))

        if detalle:
            marca = "OK " if not fallos or fallos[-1][0] != pregunta else "MAL"
            print(f"  {marca} {confianza:4.2f}  {pregunta[:46]:48} -> {obtenido}")

    total = total_intencion + total_fallback
    print()
    print(f"Preguntas de tramite/problema : {aciertos}/{total_intencion} "
          f"({aciertos / total_intencion:.0%})" if total_intencion else "")
    print(f"Fuera de tema (debe callar)   : {aciertos_fallback}/{total_fallback} "
          f"({aciertos_fallback / total_fallback:.0%})" if total_fallback else "")
    print(f"TOTAL                         : {aciertos + aciertos_fallback}/{total} "
          f"({(aciertos + aciertos_fallback) / total:.0%})")

    if fallos:
        print(f"\n{len(fallos)} fallo(s):")
        for pregunta, esperado, obtenido, conf in fallos:
            print(f"  '{pregunta}'")
            print(f"      esperado: {esperado}")
            print(f"      obtenido: {obtenido} ({conf:.2f})")


if __name__ == "__main__":
    main()
