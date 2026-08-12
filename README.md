# Chatbot para mesa de ayuda

Chatbot de apoyo para la plataforma de catastro. Responde preguntas sobre los
pasos de los tramites (cedula unica catastral, padron unico, division y fusion
de predios, avaluos, levantamiento topografico, etc.).

Clasificador de intenciones: bag-of-words + red neuronal densa (Keras),
servido con Flask.

## Requisitos

**Python 3.12.** TensorFlow todavia no publica wheels para 3.13 ni 3.14; con
esas versiones la instalacion falla.

```bash
py install 3.12
```

## Instalacion

```bash
py -3.12 -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## Entrenar el modelo

Los archivos del modelo (`words.pkl`, `classes.pkl`, `chatbot_model.keras`) **no
estan en el repositorio**: se generan a partir de `intents_spanish.json`. Hay que
entrenar antes de levantar la aplicacion por primera vez, y cada vez que se
modifiquen las intenciones.

```bash
cd chatbot
python training_chatbot.py
```

El script valida el dataset antes de entrenar y avisa de etiquetas repetidas,
patrones duplicados con etiquetas distintas e intenciones sin patrones.

## Levantar la aplicacion

```bash
cd chatbot
python app.py
```

Abre <http://127.0.0.1:5000>.

Variables de entorno opcionales:

| Variable | Por defecto | Descripcion |
|---|---|---|
| `HOST` | `127.0.0.1` | Interfaz de escucha |
| `PORT` | `5000` | Puerto |
| `FLASK_DEBUG` | `0` | `1` activa el depurador. **Solo en local**: permite ejecutar codigo arbitrario |
| `GOOGLE_CREDENTIALS_PATH` | *(sin valor)* | Ruta al JSON de la cuenta de servicio de Google |
| `GOOGLE_SHEET_NAME` | `tickets` | Nombre de la hoja de calculo |

## Estructura

```
chatbot/
  preprocess.py          Normalizacion de texto y rutas (compartido)
  training_chatbot.py    Entrena y guarda el modelo
  chatbot.py             Carga el modelo y resuelve la respuesta
  app.py                 Servidor Flask
  intents_spanish.json   Dataset de intenciones
  templates/index.html   Interfaz de chat
```

`preprocess.py` es compartido a proposito: el texto **debe** limpiarse igual al
entrenar y al predecir. Si se cambia la normalizacion, hay que reentrenar.

## Menu de tramites

El bot muestra los tramites como **botones**: el usuario hace clic y el
frontend envia la etiqueta del boton tal cual (`División de Predios`), que el
backend reconoce por coincidencia exacta.

Son botones y no una lista numerada a proposito: las respuestas ya son listas
de pasos numerados (`1. Ingresar al módulo...`), asi que pedir al usuario que
escriba un numero era ambiguo — un `1` podia significar «la opcion 1 del menu»
o «el paso 1 de este tramite». Con botones esa ambiguedad desaparece.

Se llega al menu escribiendo `menu` (tambien `opciones`, `ayuda`, `inicio`,
`volver`, `regresar`, `tramites`) o con el boton «Ver el menú», que acompaña la
bienvenida y cada respuesta. Escribir un numero suelto sigue funcionando como
atajo, aunque la interfaz ya no lo pide.

El menu **no pasa por el modelo**: elegir una opcion de una lista es una
instruccion exacta, no algo que deba adivinar el clasificador. Si se deja al
modelo, un `menu` termina en la intencion que mas se le parezca (caia en
`despedida`).

Cuando el clasificador no entiende una pregunta, la respuesta de fallback
tambien trae los botones del menu, para que el usuario tenga una salida.

Las opciones se definen en la clave `menu` de `intents_spanish.json`. El orden
define el numero que ve el usuario:

```json
"menu": [
    {"label": "Emisión de Cédula Única Catastral", "tag": "cedula_catastral_pasos"}
]
```

Cada `tag` debe existir en `intents`; `training_chatbot.py` lo valida y avisa si
alguna opcion apunta a una etiqueta inexistente. Agregar o reordenar opciones
no requiere reentrenar.

## Como agregar o mejorar respuestas

1. **Mide antes**: `python evaluar.py` y anota el resultado.
2. Edita `intents_spanish.json`. Agrega varias formas de preguntar lo mismo en
   `patterns`; con una sola el modelo memoriza y no generaliza.
3. `python training_chatbot.py`
4. **Mide despues**: `python evaluar.py`. Si bajo, revierte.

Dos lecciones que salieron de hacerlo:

- **Mas patrones no siempre es mejor.** Agregar verbos genericos ("obtener",
  "conseguir") sin el sustantivo que distingue el tramite hizo que las
  intenciones de entrada dominaran y bajo la evaluacion de 100% a 91%. Cada
  patron debe incluir la palabra que identifica al tramite.
- **Cuidado con las intenciones redundantes.** Si dos intenciones responden
  casi lo mismo, el modelo tiene que partir un pelo y falla sin que el usuario
  gane nada. Conviene fusionarlas.

## Evaluacion

```bash
python evaluar.py            # resumen
python evaluar.py --detalle  # caso por caso
```

Los casos viven en `evaluacion.json`: preguntas redactadas como las escribiria
un usuario, **ninguna copiada de los patrones de entrenamiento**. Esa es la
diferencia con el acierto que reporta `training_chatbot.py`, que es sobre sus
propios patrones y siempre da ~100% porque mide memorizacion.

Estado actual: **98-99%** sobre 82 casos (varia un par de puntos entre
entrenamientos por el barajado aleatorio y el dropout, asi que conviene mirar
el rango de varias corridas y no un solo numero).

Historial: 45% al empezar. 97% tras ampliar los patrones de los tramites
originales. 79% al incorporar las 42 intenciones de traslado de dominio,
alta de predio y cartografia (venian con un patron cada una). 98-99% tras
ampliarlas tambien.

Al agregar casos nuevos, no uses frases que ya esten en `patterns`: seria
entrenar sobre el examen y el resultado dejaria de significar nada.

## Escalamiento a ticket

El menu resuelve el "como hago X". Los tickets atienden la otra mitad: un
problema que el bot no puede resolver y que necesita a una persona.

### Cuando se ofrece

| Situacion | Que pasa |
|---|---|
| El usuario elige **Reportar un problema** (menu o bienvenida) | Inicia el formulario |
| Escribe "tengo un problema", "levantar un ticket", "hablar con un agente"... | Inicia el formulario |
| El bot no entiende una pregunta | Ofrece el boton junto al fallback |
| No entiende **dos veces seguidas** | Propone abiertamente escalar |

### El formulario

Son 7 preguntas, definidas en la clave `ticket.campos` de
`intents_spanish.json`. Se pueden agregar, quitar o reordenar sin tocar codigo:

```json
{"id": "municipio", "requerido": true, "pregunta": "¿De qué municipio es el caso?"}
```

- `requerido: false` muestra un boton **Omitir**.
- `tipo: "correo"` valida el formato y vuelve a preguntar si no es valido.
- El usuario puede escribir **cancelar** en cualquier momento.

Mientras el formulario esta abierto, **todo lo que escriba el usuario alimenta
la pregunta en turno**: no pasa por el clasificador ni por el menu. Sin esto,
responder "División de Predios" a la pregunta del tramite abriria el menu en
vez de guardar el dato.

Esto requiere memoria de conversacion: el estado vive en la sesion de Flask
(`app.py`) y se pasa a `responder(mensaje, estado)` en cada turno.

### Donde se guardan

El respaldo local **siempre** se escribe primero, y solo despues se intenta
Google Sheets. Asi un fallo de credenciales, de red o de cuota no pierde el
reporte del usuario.

- Local: `chatbot/tickets.csv` (esta en `.gitignore`: **lleva datos personales**).
- Google Sheets: solo si `GOOGLE_CREDENTIALS_PATH` apunta al JSON de la cuenta
  de servicio. Comparte la hoja con el correo de esa cuenta.

Folio con formato `MAI-AAAAMMDD-NNN`, reiniciando la numeracion cada dia. El
calculo va protegido con un candado para que dos usuarios simultaneos no
obtengan el mismo folio.

### Variable de sesion

`SECRET_KEY` firma la cookie de sesion. Si no se define, se genera una temporal
al arrancar y cada reinicio corta las conversaciones en curso. **Definela antes
de desplegar.**

## Atencion de problemas

El orden es: **primero se intenta resolver, el ticket es el ultimo recurso.**

1. Una frase vaga ("tengo un problema", "necesito ayuda") **no** abre el
   formulario: el bot pide el detalle para poder intentar resolverlo.
2. Con el detalle, busca una respuesta en las intenciones `problema_*`.
3. Toda respuesta a un problema incluye el boton **Reportar un problema**, por
   si no resolvio el caso.
4. Solo una peticion explicita ("levantar un ticket", "hablar con un agente")
   o dos fallos seguidos llevan al formulario.

Las intenciones `problema_*` cubren lo que esta documentado en los mapeos:

| Intencion | Atiende |
|---|---|
| `problema_tramite_rechazado` | Rechazo por falta de subsanacion |
| `problema_expediente_regresado` | Devolucion a ventanilla, campo o actualizacion |
| `problema_documentacion_observada` | Documentacion incorrecta o incompleta |
| `problema_no_encuentro_tramite` | Busqueda por numero de folio |
| `problema_tramite_detenido` | Tramite sin avanzar |
| `problema_propietario_no_aparece` | Propietario inexistente en la base |
| `problema_predio_sin_geometria` | Predio sin geometria en cartografia |
| `problema_construcciones_no_coinciden` | Construcciones dibujadas vs declaradas |
| `problema_tecnico_sistema` | Fallas de plataforma: **no las resuelve, escala** |

`problema_tecnico_sistema` responde con honestidad que no tiene un
procedimiento documentado y ofrece levantar el reporte. **No inventar
soluciones tecnicas**: no hay documentacion de esas fallas en el repositorio.
Si el area de sistemas aporta los procedimientos reales, se agregan como
intenciones nuevas y dejan de escalar.

El prefijo `problema_` no es cosmetico: `chatbot.py` lo usa para decidir si
adjunta el boton de escalamiento a la respuesta.
