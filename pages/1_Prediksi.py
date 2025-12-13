import streamlit as st
import joblib

st.set_page_config(
    page_title="Prediksi Sentimen",
    page_icon="🔍"
)

st.title("🔍 Prediksi Sentimen Ulasan Kopi Kenangan")

# Sidebar
with st.sidebar:
    st.markdown("## ☕ Kopi Kenangan")
    st.markdown("### Analisis Sentimen")

    st.markdown("---")

    st.markdown("ℹ️ **Tentang Aplikasi**")
    st.caption(
        "Aplikasi ini menganalisis sentimen "
        "ulasan pelanggan Kopi Kenangan "
    )

    st.markdown("---")

    st.caption("👩‍💻 Kelompok 6")
    st.caption("📚 Data Mining")

# Load TF-IDF vectorizer dan model-model
tfidf = joblib.load("tfidf.pkl")
rf_model = joblib.load("rf_model.pkl")
nb_model = joblib.load("nb_model.pkl")
svm_model = joblib.load("svm_model.pkl")

def preprocess(text):
    text = text.lower()
    return text

text_input = st.text_area("Masukkan ulasan:")

text_input = preprocess(text_input)
text_vec = tfidf.transform([text_input])

# Pilih model yang dipakai
model_choice = st.selectbox(
    "Pilih model",
    ("Random Forest", "Naive Bayes", "SVM")
)

if st.button("Prediksi"):
    if not text_input.strip():
        st.warning("Teks tidak boleh kosong")
    else:
        # Transform teks
        text_vec = tfidf.transform([text_input])

        # Pilih model
        if model_choice == "Random Forest":
            pred = rf_model.predict(text_vec)[0]
        elif model_choice == "Naive Bayes":
            pred = nb_model.predict(text_vec)[0]
        else:
            pred = svm_model.predict(text_vec)[0]

        # Tampilkan hasil
        st.write(f"Model: **{model_choice}**")
        label_map = {
            "positif": "Positif 😊",
            "negatif": "Negatif 😞"
        }

        st.subheader("Hasil Prediksi")
        if pred == "positif":
            st.success(label_map[pred])
        else:
            st.error(label_map[pred])