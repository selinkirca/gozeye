import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import plotly.graph_objects as go
import h5py

# 1. SAYFA YAPILANDIRMASI (Selin Kırca - 220706005)
st.set_page_config(page_title="Oculus AI | Göz Hastalığı Teşhis", layout="wide")

# --- GELİŞMİŞ DARK MODE CSS ---
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    section[data-testid="stSidebar"] { background-color: #161b22; border-right: 1px solid #30363d; }
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #38444d; padding: 20px; border-radius: 12px; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (GÜVENLİ MOD)
MODEL_PATH = 'eye_disease_final_mobilenet_v1.h5'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Dosya bulunamadı: {MODEL_PATH}")
        return None

    # H5 Dosya Bütünlük Kontrolü
    try:
        with h5py.File(MODEL_PATH, 'r') as f:
            # Dosya boş mu veya config eksik mi bakıyoruz
            if len(f.keys()) == 0:
                st.error("⚠️ Model dosyası boş görünüyor. Lütfen GitHub'a tekrar yükleyin.")
                return None
    except Exception as e:
        st.error(f"⚠️ Dosya okuma hatası: {e}")
        return None

    # Sürüm Yamaları
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    try:
        # compile=False ve custom_objects ile en geniş uyumluluk modunda açıyoruz
        return tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
    except Exception as e:
        st.error(f"❌ Model Yapılandırma Hatası: {e}")
        st.info("💡 İpucu: Bu hata genellikle modelin tam yüklenememesinden kaynaklanır. Modeli GitHub web arayüzünden silip tekrar yüklemeyi deneyin.")
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
    uploaded_file = st.file_uploader("Görüntü Seçiniz...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model is not None:
        img = Image.open(uploaded_file)
        enhanced = apply_clahe(img)
        st.image(enhanced, width=400)
        
        if st.button("Analizi Başlat"):
            preds = model.predict(preprocess_for_model(enhanced), verbose=0)
            idx = np.argmax(preds)
            st.success(f"Teşhis: {class_names[idx]} (%{np.max(preds)*100:.2f})")

elif menu == "📊 Metrikler":
    st.header("📈 Model Başarımı")
    st.metric("Doğruluk", "%91.4")
    st.info("MobileNetV1 mimarisi kullanılarak %91.4 genel doğruluk oranına ulaşılmıştır.")

st.divider()
st.caption("Selin Kırca - 220706005 | Giresun Üniversitesi")
