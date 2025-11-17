import pickle
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Cargar modelo
with open("model.pkl", "rb") as f:
    data = pickle.load(f)

df = data["df"]
vectorizer = data["vectorizer"]
matriz = data["matriz"]
similitud = data["similitud"]

def recomendar_para_usuario(edad, genero, categoria, presupuesto_indiv=100):
    entrada = categoria + " " + genero + " " + str(edad)
    vector = vectorizer.transform([entrada])
    scores = cosine_similarity(vector, matriz)[0]

    # Ordenar por similitud
    indices = scores.argsort()[::-1]

    # Devolver lista de candidatos (producto, precio)
    candidatos = []
    for idx in indices[:10]:  # top 10
        precio = df.iloc[idx]["precio"]
        if precio <= presupuesto_indiv:
            candidatos.append({
                "producto": df.iloc[idx]["producto"],
                "precio": precio
            })
    return candidatos


def asignar_con_presupuesto(perfiles, presupuesto_total):
    recomendaciones = []
    total_actual = 0

    for persona in perfiles:
        edad = persona["edad"]
        genero = persona["genero"]
        categoria = persona["categoria"]

        candidatos = recomendar_para_usuario(edad, genero, categoria)

        asignado = None

        for c in sorted(candidatos, key=lambda x: x["precio"]):
            if total_actual + c["precio"] <= presupuesto_total:
                asignado = c
                total_actual += c["precio"]
                break

        recomendaciones.append({
            "persona": persona["nombre"],
            "asignado": asignado
        })

    return {
        "presupuesto_total": presupuesto_total,
        "gastado": total_actual,
        "sobrante": presupuesto_total - total_actual,
        "recomendaciones": recomendaciones
    }


# -------------------------
# EJEMPLO DE USO
# -------------------------
perfiles = [
    {"nombre": "Jose", "edad": 40, "genero": "hombre", "categoria": "tecnologia"},
    {"nombre": "Maria", "edad": 20, "genero": "mujer", "categoria": "hogar"}
]

resultado = asignar_con_presupuesto(perfiles, presupuesto_total=100)
print(resultado)
