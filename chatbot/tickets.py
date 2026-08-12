# tickets.py
"""Registro de tickets de la mesa de ayuda.

El respaldo local en CSV se escribe SIEMPRE, y solo despues se intenta Google
Sheets. Asi un fallo de credenciales, de red o de cuota no hace que se pierda
el reporte del usuario: queda en disco y se puede subir despues.
"""
import csv
import datetime
import os
import threading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "tickets.csv")

# Orden de las columnas del registro. Los campos del formulario viven en
# intents_spanish.json; aqui solo se fija como se guardan.
COLUMNAS = [
    "folio", "fecha", "estado",
    "nombre", "correo", "telefono",
    "municipio", "tramite", "folio_tramite", "descripcion",
]

# El servidor atiende varias peticiones a la vez: sin este candado, dos
# usuarios simultaneos podrian calcular el mismo folio.
_candado = threading.Lock()


def _sheets_configurado():
    ruta = os.environ.get("GOOGLE_CREDENTIALS_PATH")
    return bool(ruta) and os.path.exists(ruta)


def _filas_existentes():
    if not os.path.exists(CSV_PATH):
        return []
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        return list(csv.reader(f))


def _siguiente_folio(filas):
    """Formato MAI-AAAAMMDD-NNN, con NNN reiniciando cada dia.

    Se cuenta sobre el CSV local y no sobre Google Sheets: el archivo local es
    la fuente de verdad y siempre esta disponible.
    """
    hoy = datetime.date.today().strftime("%Y%m%d")
    prefijo = f"MAI-{hoy}-"
    del_dia = [f for f in filas[1:] if f and f[0].startswith(prefijo)]
    return f"{prefijo}{len(del_dia) + 1:03d}"


def _escribir_csv(filas, fila):
    nuevo = not filas
    with open(CSV_PATH, "a", encoding="utf-8-sig", newline="") as f:
        escritor = csv.writer(f)
        if nuevo:
            escritor.writerow(COLUMNAS)
        escritor.writerow(fila)


def _escribir_sheets(fila):
    import gspread  # se importa aqui para no exigir la dependencia si no se usa

    cliente = gspread.service_account(filename=os.environ["GOOGLE_CREDENTIALS_PATH"])
    hoja = cliente.open(os.environ.get("GOOGLE_SHEET_NAME", "tickets")).sheet1
    if not hoja.get_all_values():
        hoja.append_row(COLUMNAS)
    hoja.append_row(fila)


def crear_ticket(datos):
    """Guarda un ticket y devuelve (folio, destinos).

    'destinos' es la lista de sitios donde quedo registrado, para poder decir
    con honestidad si llego a Google Sheets o solo al respaldo local.
    """
    with _candado:
        filas = _filas_existentes()
        folio = _siguiente_folio(filas)
        fila = [
            folio,
            datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "Nuevo",
        ] + [datos.get(col, "") for col in COLUMNAS[3:]]

        destinos = []
        _escribir_csv(filas, fila)
        destinos.append("local")

        if _sheets_configurado():
            try:
                _escribir_sheets(fila)
                destinos.append("sheets")
            except Exception as exc:
                print(f"[tickets] {folio} quedo solo en el respaldo local: {exc}")
        else:
            print(f"[tickets] {folio} guardado en {CSV_PATH} "
                  f"(GOOGLE_CREDENTIALS_PATH no configurada).")

    return folio, destinos
