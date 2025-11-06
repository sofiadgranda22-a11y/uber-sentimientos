# =========================
# 📊 Análisis de Opiniones - Uber
# =========================

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from collections import Counter
import plotly.express as px
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from transformers import pipeline
import nltk

# Descargar recursos de NLTK
nltk.download('stopwords')
nltk.download('wordnet')

# Configuración inicial
st.set_page_config(page_title="Análisis de Sentimientos", layout="wide")
st.title("🚗 Análisis de Opiniones de Clientes - Uber")

st.write("Sube un archivo CSV con una columna que contenga las opiniones de los usuarios (por ejemplo, 'opinion', 'comentario' o 'review').")

# --- Carga del archivo ---
uploaded_file = st.file_uploader("📂 Cargar archivo CSV", type=["csv"])

if uploaded_file is not None:
    # Leer CSV con tolerancia a errores de formato
    try:
        df = pd.read_csv(uploaded_file, encoding='latin-1', on_bad_lines='skip')
    except Exception:
        df = pd.read_csv(uploaded_file, encoding='utf-8', on_bad_lines='skip')

    df.dropna(inplace=True)
    st.success(f"Archivo cargado con {len(df)} registros.")
    st.write("Vista previa de los datos:")
    st.write(df.head())

    # --- Detección automática de la columna de texto ---
    columna_texto = None
    candidatas = ['text', 'texto', 'opinion', 'opinión', 'comentario', 'comentarios', 'review', 'review_text']

    for c in df.columns:
        cname = c.lower().strip()
        if cname in candidatas:
            columna_texto = c
            break

    # Si no se encontró coincidencia exacta, buscar aproximada
    if columna_texto is None:
        for c in df.columns:
            if any(k in c.lower() for k in ['text', 'opinion', 'coment', 'review']):
                columna_texto = c
                break

    # Si no se encuentra, mostrar error
    if columna_texto is None:
        st.error("⚠️ No se encontró una columna con opiniones. Asegúrate de que tu CSV tenga una columna llamada 'opinion', 'comentario' o similar.")
        st.write("Columnas encontradas:", list(df.columns))
        st.stop()
    else:
        st.info(f"Usando la columna: **{columna_texto}** como campo de texto para el análisis.")

    # --- LIMPIEZA DE TEXTO ---
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('spanish'))

    def limpiar_texto(texto):
        palabras = texto.lower().split()
        palabras = [p for p in palabras if p.isalpha() and p not in stop_words]
        palabras = [lemmatizer.lemmatize(p) for p in palabras]
        return " ".join(palabras)

    df["texto_limpio"] = df[columna_texto].astype(str).apply(limpiar_texto)

    # --- FRECUENCIA DE PALABRAS ---
    todas_palabras = " ".join(df["texto_limpio"]).split()
    contador = Counter(todas_palabras)
    top_palabras = pd.DataFrame(contador.most_common(10), columns=["Palabra", "Frecuencia"])

    # --- GRÁFICO DE BARRAS ---
    st.subheader("🔠 Palabras más frecuentes")
    fig_bar = px.bar(top_palabras, x="Palabra", y="Frecuencia",
                     title="Top 10 palabras más frecuentes (tras limpieza)",
                     color="Frecuencia", color_continuous_scale="Blues")
    st.plotly_chart(fig_bar, use_container_width=True)

    # --- NUBE DE PALABRAS ---
    st.subheader("☁️ Nube de palabras")
    wordcloud = WordCloud(width=900, height=400, background_color='white').generate(" ".join(todas_palabras))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis("off")
    st.pyplot(fig)

    # --- ANÁLISIS DE SENTIMIENTOS ---
    st.subheader("🔍 Clasificación de sentimientos")
    clasificador = pipeline("sentiment-analysis", model="cardiffnlp/twitter-xlm-roberta-base-sentiment")

    def traducir_sentimiento(label):
        if "positive" in label.lower():
            return "Positivo"
        elif "negative" in label.lower():
            return "Negativo"
        else:
            return "Neutro"

    resultados = []
    for texto in df[columna_texto]:
        try:
            pred = clasificador(texto[:512])[0]
            sentimiento = traducir_sentimiento(pred["label"])
            resultados.append(sentimiento)
        except Exception:
            resultados.append("Neutro")

    df["Sentimiento"] = resultados

    # --- Mostrar resultados ---
    st.subheader("📋 Resultados del modelo")
    st.dataframe(df[[columna_texto, "Sentimiento"]])

    # --- GRÁFICO DE SENTIMIENTOS ---
    conteo = df["Sentimiento"].value_counts().reset_index()
    conteo.columns = ["Sentimiento", "Cantidad"]

    fig_sent = px.pie(
        conteo,
        names="Sentimiento",
        values="Cantidad",
        title="📊 Distribución de sentimientos (Positivo / Negativo / Neutro)",
        color="Sentimiento",
        color_discrete_map={"Positivo": "#4CAF50", "Negativo": "#E74C3C", "Neutro": "#F1C40F"}
    )
    st.plotly_chart(fig_sent, use_container_width=True)

        # --- GRÁFICO ADICIONAL: Longitud del comentario vs Sentimiento ---
    st.subheader("📈 Relación entre longitud del comentario y sentimiento")

    # Calcular longitud del comentario (en palabras)
    df["Longitud"] = df[columna_texto].astype(str).apply(lambda x: len(x.split()))

    # Gráfico de caja (boxplot) para ver la distribución
    fig_len = px.box(
        df,
        x="Sentimiento",
        y="Longitud",
        color="Sentimiento",
        title="Distribución de longitud de comentarios por tipo de sentimiento",
        color_discrete_map={"Positivo": "#4CAF50", "Negativo": "#E74C3C", "Neutro": "#F1C40F"}
    )
    st.plotly_chart(fig_len, use_container_width=True)

    # Promedio de longitud por sentimiento
    promedio_long = df.groupby("Sentimiento")["Longitud"].mean().reset_index().round(1)
    st.write("Promedio de palabras por tipo de sentimiento:")
    st.dataframe(promedio_long)


    # --- NUEVO COMENTARIO ---
    st.subheader("✍️ Analiza una nueva opinión")
    nueva_opinion = st.text_area("Escribe aquí tu comentario sobre el servicio de Uber:")

    if st.button("Analizar sentimiento"):
        if nueva_opinion.strip():
            resultado = clasificador(nueva_opinion)[0]
            sentimiento = traducir_sentimiento(resultado["label"])
            st.success(f"**Sentimiento detectado:** {sentimiento}")
        else:
            st.warning("Por favor, escribe un comentario antes de analizar.")
