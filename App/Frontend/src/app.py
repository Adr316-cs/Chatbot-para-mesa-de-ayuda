from flask import Flask, render_template


app = Flask(__name__)

@app.route("/contenido")
def contenido():
    return """
    <h2>Contenido generado por Python</h2>
    <p>Este texto viene desde app.py</p>
    """
#if __name__ == '__main__':
#    app.run(debug=True)
app.run(debug=True)