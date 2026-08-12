# reporte.py
"""Resume la bitacora de preguntas no entendidas.

Uso:  python reporte.py [--dias N] [--top N] [--todo]

Agrupa las preguntas parecidas en vez de listarlas crudas: 'no puedo entrar' y
'no puedo entrar al sistema' son la misma necesidad y deben contarse juntas,
porque lo que importa es cuanta gente pregunta lo mismo.

Como usarlo: lo que aparezca arriba con varias repeticiones es lo siguiente
que conviene agregar a intents_spanish.json.
"""
import argparse
import collections
import csv
import datetime
import os
import sys

from preprocess import clean_up_sentence, ensure_nltk_data, palabras_vacias
from registro import CSV_PATH


def cargar(dias=None):
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        filas = list(csv.DictReader(f))
    if dias:
        corte = datetime.datetime.now() - datetime.timedelta(days=dias)
        filas = [r for r in filas
                 if datetime.datetime.strptime(r["fecha"], "%Y-%m-%d %H:%M:%S") >= corte]
    return filas


# Cuanto se deben parecer dos preguntas para contarse como la misma necesidad.
# Se compara por palabras con contenido: 'como pago el predial' y 'donde pago
# el impuesto predial' comparten {pag, predial} y quedan juntas.
PARECIDO_MINIMO = 0.5


def _contenido(texto):
    return set(clean_up_sentence(texto)) - palabras_vacias()


def agrupar(filas):
    """Junta las preguntas que expresan la misma necesidad.

    Agrupar por conjunto exacto de raices no sirve: cualquier palabra de mas
    crea un grupo nuevo y todo aparece como '1x'. Se agrupa por parecido, con
    asignacion voraz al primer grupo compatible.
    """
    grupos = []  # [(conjunto_representativo, [filas])]
    for r in filas:
        clave = _contenido(r["mensaje"])
        if not clave:
            clave = {r["mensaje"].strip().lower()}
        for representante, miembros in grupos:
            union = representante | clave
            if union and len(representante & clave) / len(union) >= PARECIDO_MINIMO:
                miembros.append(r)
                break
        else:
            grupos.append((clave, [r]))
    return sorted((m for _, m in grupos), key=len, reverse=True)


def mostrar(titulo, filas, top, con_candidato):
    print(f"\n{titulo}: {len(filas)}")
    if not filas:
        return
    grupos = agrupar(filas)
    print(f"  ({len(grupos)} preguntas distintas)\n")
    for grupo in grupos[:top]:
        # El mensaje mas largo del grupo suele ser el mas descriptivo.
        ejemplo = max((r["mensaje"] for r in grupo), key=len)
        print(f"  {len(grupo):3}x  {ejemplo[:66]}")
        if con_candidato:
            respuestas = collections.Counter(r["mejor_candidato"] for r in grupo)
            for cand, n in respuestas.most_common(2):
                confs = [float(r["confianza"]) for r in grupo if r["mejor_candidato"] == cand]
                print(f"          respondio: {cand[:44]} ({n}x, conf. media "
                      f"{sum(confs)/len(confs):.2f})")
        elif len(grupo) > 1:
            otras = {r["mensaje"] for r in grupo} - {ejemplo}
            for o in list(otras)[:2]:
                print(f"          tambien: {o[:56]}")
    if len(grupos) > top:
        print(f"\n  ... y {len(grupos) - top} preguntas distintas mas "
              f"(usa --top {len(grupos)} para verlas todas)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dias", type=int, help="solo los ultimos N dias")
    ap.add_argument("--top", type=int, default=15, help="cuantos grupos mostrar")
    ap.add_argument("--todo", action="store_true", help="sin limite de grupos")
    args = ap.parse_args()

    ensure_nltk_data()
    filas = cargar(args.dias)

    if not filas:
        print(f"No hay registros en {CSV_PATH}.")
        print("Se llena solo conforme la gente use el chatbot.")
        return 0

    top = 10**6 if args.todo else args.top
    periodo = f" (ultimos {args.dias} dias)" if args.dias else ""
    print(f"Bitacora de preguntas no entendidas{periodo}")
    print(f"Total registrado: {len(filas)}")

    sin = [r for r in filas if r["motivo"] == "sin_coincidencia"]
    baja = [r for r in filas if r["motivo"] == "confianza_baja"]

    mostrar("NO ENTENDIDAS  (respondio el fallback; son las mas urgentes)",
            sin, top, con_candidato=False)
    mostrar("CONTESTO DUDANDO  (revisa si la respuesta fue la correcta)",
            baja, top, con_candidato=True)

    print("\nQue hacer con esto:")
    print("  1. Lo de arriba con mas repeticiones es lo siguiente que hay que cubrir.")
    print("  2. Si ya existe una intencion que responde eso, agregale esas frases")
    print("     como patrones. Si no existe, creala.")
    print("  3. python training_chatbot.py && python evaluar.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
