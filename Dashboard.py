import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from collections import Counter
import re

# Load TF-IDF vectorizer dan model-model
tfidf = joblib.load("tfidf.pkl")
rf_model = joblib.load("rf_model.pkl")
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")

def preprocess(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)   # hapus URL
    text = re.sub(r"[^a-zA-Z\s]", "", text)     # hapus simbol
    text = re.sub(r"\s+", " ", text).strip()
    return text

st.set_page_config(
    page_title="Dashboard Sentimen Kopi Kenangan",
    page_icon="☕",
    layout="wide"
)

COLOR_MAP = {
    "Positif": "#8ecbff",   # biru muda
    "Negatif": "#ff6b6b"    # merah lembut
}

st.title("📊 Dashboard Sentimen Ulasan Kopi Kenangan")

# Sidebar
with st.sidebar:
    st.markdown("## ☕ Kopi Kenangan")
    st.markdown("### Analisis Sentimen Ulasan")

    st.markdown("---")

    st.markdown("ℹ️ **Tentang Aplikasi**")
    st.caption(
        "Aplikasi ini menganalisis sentimen "
        "ulasan pelanggan Kopi Kenangan "
    )

    st.markdown("---")

    st.caption("👩‍💻 Kelompok 6")
    st.caption("📚 Data Mining")

# Upload Dataset
st.sidebar.markdown("### 📂 Upload Dataset")

uploaded_file = st.sidebar.file_uploader(
    "Upload file CSV",
    type=["csv"]
)

# Load Data
@st.cache_data
def load_data(uploaded_file=None):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    return pd.read_csv("sentiment.csv")

df = load_data(uploaded_file)

# Validasi kolom
if "content" not in df.columns:
    st.error("Dataset harus memiliki kolom 'content'")
    st.stop()

# Prediksi Otomatis Dataset Upload
df["content_clean"] = df["content"].astype(str).apply(preprocess)

X = tfidf.transform(df["content_clean"])
df["predicted_label"] = rf_model.predict(X)



# KPI
total_ulasan = len(df)
total_positif = (df["label"] == "positif").sum()
total_negatif = (df["label"] == "negatif").sum()

persen_positif = total_positif / total_ulasan * 100

col1, col2, col3, col4 = st.columns(4)
col1.metric("☕ Total Ulasan", total_ulasan)
col2.metric("😊 Positif", total_positif)
col3.metric("😞 Negatif", total_negatif)
col4.metric("📈 % Positif", f"{persen_positif:.1f}%")

st.markdown("### 📊 Analisis Sentimen")

col1, col2 = st.columns([1, 1])

# Pie Chart Distribusi Sentimen
with col1:
    labels = ["Positif", "Negatif"]
    sizes = [total_positif, total_negatif]
    colors = [COLOR_MAP[label] for label in labels]

    fig, ax = plt.subplots(figsize=(2, 2), facecolor="none")
    ax.set_facecolor("none")

    ax.pie(
        sizes,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"color": "white", "fontsize": 5}
    )

    ax.axis("equal")
    st.pyplot(fig, transparent=True)
    

# Bar Chart Jumlah Ulasan Per Sentimen
with col2:
    st.markdown("### 📊 Jumlah Ulasan per Sentimen")

    sentiment_count = df["label"].value_counts()

    st.bar_chart(sentiment_count)

st.info(
        f"Mayoritas ulasan bersentimen positif ({persen_positif:.1f}%), "
        "menunjukkan persepsi pelanggan terhadap Kopi Kenangan cenderung baik."
    )

# Top 10 Kata Positif & Negatif
def get_top_words(texts, n=10):
    words = []
    for text in texts:
        text = re.sub(r"[^\w\s]", "", text.lower())
        words.extend(text.split())
    return Counter(words).most_common(n)

col1, col2 = st.columns(2)

with col1:
    st.markdown("### 👍 Top 10 Kata Positif")
    top_pos = get_top_words(df[df["label"]=="positif"]["content"])
    st.dataframe(pd.DataFrame(top_pos, columns=["Kata", "Frekuensi"]))

with col2:
    st.markdown("### 👎 Top 10 Kata Negatif")
    top_neg = get_top_words(df[df["label"]=="negatif"]["content"])
    st.dataframe(pd.DataFrame(top_neg, columns=["Kata", "Frekuensi"]))

# Frekuensi Kata Teratas
st.markdown("### 📊 Frekuensi Kata Teratas")

all_words = df["content"].str.lower().str.split().explode()
top_words = all_words.value_counts().head(15)

st.bar_chart(top_words)

# Panjang Ulasan
df["panjang_teks"] = df["content"].astype(str).apply(len)

st.markdown("### 📝 Rata-rata Panjang Ulasan per Sentimen")
st.bar_chart(df.groupby("label")["panjang_teks"].mean())

# Tabel Insight
st.markdown("### 📋 Ringkasan Statistik")

summary = df.groupby("label").agg(
    Jumlah_Ulasan=("content", "count"),
    Panjang_Rata=("panjang_teks", "mean")
)

st.dataframe(summary)



