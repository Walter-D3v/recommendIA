from flask import Flask, request, jsonify
import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# --- Cargar modelo ---
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
vectorizer = data["vectorizer"]
matriz = data["matriz"]
similitud = data["similitud"]

# --- Recomendación individual usando intereses ---
def recomendar_para_usuario(edad, genero, categoria, interes):
    entrada = categoria + " " + genero + " " + interes + " " + str(edad)
    vector = vectorizer.transform([entrada])
    scores = cosine_similarity(vector, matriz)[0]

    indices = scores.argsort()[::-1]

    candidatos = []
    for idx in indices[:10]:
        candidatos.append({
            "producto": df.iloc[idx]["producto"],
            "precio": int(df.iloc[idx]["precio"]),
            "interes": df.iloc[idx]["interes"]
        })

    return candidatos


# --- Asignación con presupuesto ---
def asignar_con_presupuesto(perfiles, presupuesto_total):
    recomendaciones = []
    total_actual = 0

    for persona in perfiles:
        candidatos = recomendar_para_usuario(
            persona["edad"],
            persona["genero"],
            persona["categoria"],
            persona["interes"]
        )

        asignado = None

        # Seleccionar el más barato dentro de los recomendados
        for c in sorted(candidatos, key=lambda x: x["precio"]):
            if total_actual + c["precio"] <= presupuesto_total:
                asignado = {
                    "producto": c["producto"],
                    "precio": int(c["precio"]),
                    "interes": c["interes"]
                }
                total_actual += int(c["precio"])
                break

        recomendaciones.append({
            "persona": persona["nombre"],
            "asignado": asignado
        })

    return {
        "presupuesto_total": int(presupuesto_total),
        "gastado": int(total_actual),
        "sobrante": int(presupuesto_total - total_actual),
        "recomendaciones": recomendaciones
    }


# --- Endpoint multiple ---
@app.route("/recomendacion/multiple", methods=["POST"])
def recomendacion_multiple():
    data = request.json
    presupuesto = data["presupuesto_total"]
    perfiles = data["perfiles"]

    resultado = asignar_con_presupuesto(perfiles, presupuesto)
    return jsonify(resultado)


# --- Endpoint individual ---
@app.route("/recomendacion", methods=["POST"])
def recomendacion():
    datos = request.json

    candidatos = recomendar_para_usuario(
        datos["edad"],
        datos["genero"],
        datos["categoria"],
        datos["interes"]
    )

    return jsonify({"recomendaciones": candidatos[:3]})


if __name__ == "__main__":
    app.run(debug=True)
