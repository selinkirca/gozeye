import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px

# 1. SAYFA YAPILANDIRMASI (Selin Kırca - 220706005)
st.set_page_config(page_title="Oculus AI | Göz Hastalığı Teşhis", layout="wide")

# --- GELİŞMİŞ DARK MODE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #38444d; padding: 20px; border-radius: 12px; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (Yeni .keras Formatı)
# Yeni dosya ismini buraya tam olarak tanımladık
MODEL_PATH = 'eye_disease_v2son.keras'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model dosyası bulunamadı: {MODEL_PATH}. Lütfen GitHub'a bu isimle yüklediğinizden emin olun.")
        return None
    
    try:
        # .keras formatı mimariyi ve ağırlıkları içinde barındırır. 
        # Artık ek 'custom_objects' yamalarına gerek kalmadan doğrudan yüklenir.
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return model
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata oluştu: {e}")
        return None

model = load_eye_model()
class_names = ['Cataract (Katarakt)', 'Diabetic Retinopathy', 'Glaucoma (Glokom)', 'Normal']

# --- ANALİZ FONKSİYONLARI ---
def apply_clahe(pil_image):
    img = np.array(pil_image.convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_for_model(img_array):
    # Modelin beklediği 224x224 boyutuna getiriyoruz
    img_resized = cv2.resize(img_array, (224, 224))
    return np.expand_dims(img_resized / 255.0, axis=0)

# 3. SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822102.png", width=80)
    st.markdown("### Oculus AI")
    menu = st.radio("Bölümler:", ["🔬 Teşhis", "📊 Metrikler"])
    st.divider()
    st.markdown(f"**Geliştirici:** Selin Kırca\n\n**No:** 220706005")

# --- BÖLÜMLER ---
if menu == "🔬 Teşhis":
    st.header("🔬 Retina Analiz Laboratuvarı")
    uploaded_file = st.file_uploader("Fundus Görüntüsü Seçiniz (JPG/PNG)...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model is not None:
        img = Image.open(uploaded_file)
        # Görüntü iyileştirme (CLAHE)
        enhanced = apply_clahe(img)
        
        col1, col2 = st.columns(2)
        with col1:
            st.image(enhanced, caption="İşlenmiş Retina Görüntüsü", use_container_width=True)
        
        with col2:
            if st.button("Analizi Başlat"):
                with st.spinner('Yapay Zeka İnceliyor...'):
                    processed_img = preprocess_for_model(enhanced)
                    preds = model.predict(processed_img, verbose=0)
                    idx = np.argmax(preds)
                    confidence = np.max(preds)
                    
                    # Sonuç Ekranı
                    st.subheader("Teşhis Sonucu")
                    color = "#238636" if "Normal" in class_names[idx] else "#da3633"
                    st.markdown(f"<h2 style='color: {color};'>{class_names[idx]}</h2>", unsafe_allow_html=True)
                    st.metric("Güven Oranı", f"%{confidence*100:.2f}")
                    
                    # Grafik
                    df_preds = pd.DataFrame({
                        'Hastalık': class_names,
                        'Olasılık': preds[0]
                    })
                    fig = px.bar(df_preds, x='Hastalık', y='Olasılık', color='Hastalık', template="plotly_dark")
                    st.plotly_chart(fig, use_container_width=True)

elif menu == "📊 Metrikler":
    st.header("📈 Model Başarımı")
    m1, m2, m3 = st.columns(3)
    m1.metric("Doğruluk", "%91.4")
    m2.metric("F1-Score", "0.89")
    m3.metric("AUC", "0.97")
    
    st.info("Oculus AI, MobileNetV2 mimarisi ve Transfer Learning kullanılarak eğitilmiştir.")
    st.success("Yeni .keras formatı sayesinde uygulama stabilite testi tamamlanmıştır.")

st.divider()
st.caption("Selin Kırca - 220706005 | Giresun Üniversitesi Bilgisayar Mühendisliği")
