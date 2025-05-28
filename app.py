import streamlit as st
import joblib

# -----------------------------
# Load model dan vectorizer
# -----------------------------
model = joblib.load('best_logistic_model.pkl')
tfidf = joblib.load('tfidf.pkl')

# -----------------------------
# Label dibalik: 0 = Positif, 1 = Negatif
# -----------------------------
label_map = {
    0: "Sentimen Positif",
    1: "Sentimen Negatif"
}

# -----------------------------
# Desain halaman
# -----------------------------
st.set_page_config(page_title="Prediksi Sentimen Twitter", page_icon="🐦", layout="centered")

# -----------------------------
# Sidebar - Profil Pembuat
# -----------------------------
st.sidebar.image("Azhar.jpg", width=100)  # Ganti URL dengan foto profilmu jika ada
st.sidebar.markdown("""
## Profil Pembuat
**Nama:** Azhar Zuhro  
**Email:** azharzuhro74@gmail.com.com  
**GitHub:** [github.com/azhrzuhro](https://github.com/azhrzuhro)  
**LinkedIn:** [linkedin.com/in/azhar-zuhro](https://www.linkedin.com/in/azhar-zuhro)
""")

# -----------------------------
# Logo dan Judul
# -----------------------------
st.markdown(
    """
    <div style="text-align: center;">
        <img src="https://cdn-icons-png.flaticon.com/512/733/733579.png" width="80"/>
        <h1 style="color: #1DA1F2;">Analisis Sentimen Twitter Tentang Kecanduan Judi Online</h1>
        <p>Prediksi sentimen dari cuitan atau teks secara otomatis</p>
    </div>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Input pengguna
# -----------------------------
st.markdown("### Masukkan Teks:")
user_input = st.text_area("Contoh: Aku sangat senang dengan layanan ini!", height=150)

# -----------------------------
# Tombol Prediksi
# -----------------------------
if st.button("🔍 Prediksi Sentimen"):
    if user_input.strip() == "":
        st.warning("⚠️ Tolong masukkan teks terlebih dahulu.")
    else:
        input_vector = tfidf.transform([user_input])
        prediction = model.predict(input_vector.toarray())[0]
        prediction_text = label_map.get(prediction, "Label tidak dikenal")

        # Gaya warna hasil prediksi
        if prediction == 0:
            st.success(f"**{prediction_text}**")
        else:
            st.error(f"**{prediction_text}**")

# -----------------------------
# Footer
# -----------------------------
st.markdown("""
<hr>
<div style='text-align: center; color: grey;'>
    Dibuat untuk analisis teks media sosial<br>
    Created by Azhar Zuhro
</div>
""", unsafe_allow_html=True)
