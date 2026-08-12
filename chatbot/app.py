# app.py
import os
import secrets

from flask import Flask, jsonify, render_template, request, session

from chatbot import responder

# Longitud maxima del mensaje. Evita que una peticion enorme llegue al modelo.
MAX_MENSAJE = 1000

app = Flask(__name__)

# La sesion guarda el estado de la conversacion (el formulario de ticket en
# curso). Sin SECRET_KEY fija, cada reinicio invalida las sesiones abiertas:
# sirve para desarrollo, pero en produccion hay que definirla.
_clave = os.environ.get("SECRET_KEY")
if not _clave:
    _clave = secrets.token_hex(32)
    print("[app] SECRET_KEY no definida; se generó una temporal. "
          "Defínela como variable de entorno antes de desplegar.")
app.secret_key = _clave


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/favicon.ico")
def favicon():
    """El icono va embebido en el HTML, pero los navegadores piden esta ruta de
    todos modos. Se responde 204 para que no quede un 404 en el registro."""
    return "", 204


@app.route("/chat", methods=["POST"])
def chat():
    datos = request.get_json(silent=True) or {}
    mensaje = datos.get("message")

    if not isinstance(mensaje, str) or not mensaje.strip():
        return jsonify({"error": "Falta el campo 'message'."}), 400
    if len(mensaje) > MAX_MENSAJE:
        return jsonify({"error": f"El mensaje excede {MAX_MENSAJE} caracteres."}), 400

    try:
        resultado, estado = responder(mensaje, session.get("estado"))
    except Exception:
        app.logger.exception("Error al procesar el mensaje")
        return jsonify({"error": "No se pudo procesar el mensaje."}), 500

    session["estado"] = estado
    return jsonify(resultado)


@app.route("/reiniciar", methods=["POST"])
def reiniciar():
    """Olvida la conversacion (util si el usuario se queda atorado)."""
    session.pop("estado", None)
    return jsonify({"ok": True})


if __name__ == "__main__":
    # debug=True expone el depurador de Werkzeug, que permite ejecutar codigo
    # arbitrario. Se activa solo si se pide de forma explicita.
    debug = os.environ.get("FLASK_DEBUG", "0") == "1"
    app.run(host=os.environ.get("HOST", "127.0.0.1"),
            port=int(os.environ.get("PORT", 5000)),
            debug=debug)
