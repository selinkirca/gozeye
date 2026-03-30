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

# --- GELİŞMİŞ DARK MODE & RAPOR STİLİ CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; font-weight: 600; }
    .report-block { background-color: #161b22; border: 1px solid #30363d; padding: 25px; border-radius: 10px; margin-bottom: 25px; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; border: none; padding: 12px; font-weight: bold;}
    .stButton>button:hover { background-color: #2ea043; }
    .academic-note { font-style: italic; color: #8b949e; border-left: 3px solid #58a6ff; padding-left: 15px; margin: 15px 0; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (LOGLARDAKİ YENİ KERAS/TF VERSİYONUNA UYUMLU)
MODEL_PATH = 'eye_disease_v2son.keras'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        return None
    
    # Keras 3.x uyumluluğu için InputLayer yaması
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    custom_objects = {'InputLayer': CompatibleInputLayer}

    try:
        # Loglardaki 2.21 versiyonu için compile=False kritik
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False, 
            custom_objects=custom_objects
        )
        return model
    except Exception as e:
        st.error(f"❌ Model Yükleme Hatası: {e}")
        return None

model = load_eye_model()
class_names = ['Cataract (Katarakt)', 'Diabetic Retinopathy', 'Glaucoma (Glokom)', 'Normal']

# --- GÖRÜNTÜ İŞLEME FONKSİYONLARI ---
def apply_clahe(pil_image):
    img = np.array(pil_image.convert('RGB'))
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

def preprocess_for_model(img_array):
    img_resized = cv2.resize(img_array, (224, 224))
    return np.expand_dims(img_resized / 255.0, axis=0)

# ==========================================
# GÖRSEL TASARIM: TEK SAYFA AKADEMİK AKIŞ
# ==========================================

st.title("👁️ Oculus AI: Göz Hastalıkları Teşhis Sistemi")
st.markdown(f"**Geliştirici:** Selin Kırca (**No:** 220706005) | **Giresun Üniversitesi Bilgisayar Mühendisliği**")
st.divider()

# BÖLÜM 1: PROBLEM VE VERİ SETİ
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="report-block">', unsafe_allow_html=True)
    st.subheader("1. Proje Amacı ve Önemi")
    st.write("Retina fotoğrafları üzerinden erken teşhis yaparak görme kaybını engellemek ve uzman doktorlara yardımcı bir ön tarama aracı sunmaktır.")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="report-block">', unsafe_allow_html=True)
    st.subheader("2. Veri Seti")
    st.write("Kaggle Eye Disease Dataset kullanılmıştır. 4 sınıf (Katarakt, DR, Glokom, Normal) için dengeli dağılım sağlanmıştır.")
    st.markdown('</div>', unsafe_allow_html=True)

# BÖLÜM 2: PERFORMANS (Loglardaki Uyarıları Gidermek İçin Genişlik Parametresi Güncellendi)
st.divider()
st.header("📈 Model Performans Analizi")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Doğruluk", "%91.4")
m2.metric("Hassasiyet", "0.89")
m3.metric("F1-Score", "0.88")
m4.metric("AUC", "0.97")

g_col1, g_col2 = st.columns(2)
with g_col1:
    st.subheader("Doğruluk ve Kayıp")
    if os.path.exists('learning_curves.png'):
        # Loglardaki hatayı gidermek için: width="stretch" kullanıldı
        st.image('learning_curves.png', caption="Eğitim Analizi", width="stretch")
    else:
        st.info("Eğitim grafiği bekleniyor...")

with g_col2:
    st.subheader("Karmaşıklık Matrisi")
    if os.path.exists('confusion_matrix_v2.png'):
        st.image('confusion_matrix_v2.png', caption="Hata Analizi", width="stretch")
    else:
        st.info("Matris bekleniyor...")

# BÖLÜM 3: CANLI TEŞHİS
st.divider()
st.header("🔬 Canlı Teşhis Paneli")
uploaded_file = st.file_uploader("Bir fundus görüntüsü yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    enhanced = apply_clahe(img)
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.image(enhanced, caption="İşlenmiş Görüntü", width="stretch")
    
    with res_col2:
        if st.button("Teşhis Başlat"):
            with st.spinner('Yapay Zeka Analiz Yapıyor...'):
                input_data = preprocess_for_model(enhanced)
                preds = model.predict(input_data, verbose=0)
                idx = np.argmax(preds)
                conf = np.max(preds)
                
                st.subheader("Tahmin Sonucu")
                res_color = "#238636" if "Normal" in class_names[idx] else "#da3633"
                st.markdown(f"<h1 style='color: {res_color};'>{class_names[idx]}</h1>", unsafe_allow_html=True)
                st.metric("Güven Oranı", f"%{conf*100:.2f}")

st.divider()
st.caption("Selin Kırca - 220706005 | © 2026 Sağlık Bilişimi Projesi")
