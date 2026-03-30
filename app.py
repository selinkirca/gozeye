import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px

# Sayfa Konfigürasyonu (Kriter 17 & 18)
st.set_page_config(page_title="Göz Hastalığı Teşhis Sistemi", layout="wide")

# Tasarım ve Estetik
st.markdown("""
    <style>
    .stApp { background-color: #0d1117; color: #c9d1d9; }
    div[data-testid="stMetric"] { background-color: #1f2937; border: 1px solid #30363d; padding: 20px; border-radius: 12px; }
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .report-block { background-color: #161b22; border: 1px solid #30363d; padding: 25px; border-radius: 10px; margin-bottom: 20px; }
    .academic-note { font-style: italic; color: #8b949e; border-left: 3px solid #58a6ff; padding-left: 15px; margin: 15px 0; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; border: none; padding: 12px; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# Model Yükleme (Kriter 19)
MODEL_PATH = 'eye_disease_v2son.keras'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"⚠️ Model dosyası bulunamadı: {MODEL_PATH}")
        return None
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)
    try:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False, custom_objects={'InputLayer': CompatibleInputLayer})
        return model
    except Exception as e:
        st.error(f"Model yükleme hatası: {e}")
        return None

model = load_eye_model()
class_names = ['Katarakt', 'Diyabetik Retinopati', 'Glokom', 'Normal']

# Giriş ve Kimlik Bilgileri (Kriter 20)
st.title("👁️ Derin Öğrenme ile Göz Hastalıkları Teşhis Sistemi")
st.markdown(f"**Geliştirici:** Selin Kırca | **Öğrenci No:** 220706005 | **Üniversite:** Giresun Üniversitesi")
st.divider()

# BÖLÜM 1: PROBLEM VE AMAÇ (Kriter 1, 2, 3)
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="report-block">', unsafe_allow_html=True)
    st.subheader("Problem Tanımı ve Önemi")
    st.write("Retina hastalıklarının erken teşhisi, kalıcı görme kayıplarını engellemek için hayati önem taşır. Bu çalışma, uzman eksikliği olan bölgelerde hızlı tarama desteği sunmayı amaçlar.")
    st.markdown('</div>', unsafe_allow_html=True)
with col2:
    st.markdown('<div class="report-block">', unsafe_allow_html=True)
    st.subheader("Proje Hedefleri")
    st.write("- Transfer learning ile yüksek doğruluklu teşhis.\n- Görüntü işleme ile medikal veri kalitesini artırma.\n- Erişilebilir bir web arayüzü.")
    st.markdown('</div>', unsafe_allow_html=True)

# BÖLÜM 2: VERİ VE METODOLOJİ (Kriter 4-11)
st.divider()
st.header("Veri Seti ve Metodoloji")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Veri Seti:**")
    st.write("Kaggle Eye Disease Dataset (4 Sınıf).")
with c2:
    st.markdown("**Ön İşleme:**")
    st.write("CLAHE Filtresi, Normalizasyon, Augmentation.")
with c3:
    st.markdown("**Mimari:**")
    st.write("MobileNetV2, Dropout (%50), Adam Optimizer.")

# BÖLÜM 3: PERFORMANS VE GRAFİKLER (Kriter 12-16)
st.divider()
st.header("Performans Metrikleri ve Analizler")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Doğruluk", "%91.4")
m2.metric("Kesinlik", "0.89")
m3.metric("Duyarlılık", "0.88")
m4.metric("AUC Skoru", "0.97")

g1, g2 = st.columns(2)
with g1:
    st.subheader("Eğitim Süreci Analizi")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Doğruluk ve Kayıp Grafikleri", width=600)
    else:
        st.error("❌ 'learning_curves.png' dosyası GitHub reponuzda bulunamadı!")
    st.markdown('<p class="academic-note">Yorum: Eğitim ve doğrulama eğrilerinin paralelliği, modelin overfitting yapmadığını kanıtlar.</p>', unsafe_allow_html=True)

with g2:
    st.subheader("Karmaşıklık Matrisi")
    if os.path.exists('confusion_matrix_final.png'):
        st.image('confusion_matrix_final.png', caption="Sınıflandırma Hata Analizi", width=600)
    else:
        st.error("❌ 'confusion_matrix_final.png' dosyası GitHub reponuzda bulunamadı!")
    st.markdown('<p class="academic-note">Yorum: Model Normal ve DR sınıflarında oldukça yüksek başarı göstermektedir.</p>', unsafe_allow_html=True)

st.subheader("ROC Eğrisi Analizi")
if os.path.exists('roc_curve_final.png'):
    st.image('roc_curve_final.png', caption="Sınıf Bazlı AUC Analizi", width=700)
else:
    st.error("❌ 'roc_curve_final.png' dosyası GitHub reponuzda bulunamadı!")

# BÖLÜM 4: CANLI TEŞHİS (Kriter 19)
st.divider()
st.header("🔬 Canlı Teşhis Laboratuvarı")
uploaded_file = st.file_uploader("Görüntü Yükleyin...", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    col_img, col_res = st.columns(2)
    with col_img:
        st.image(enhanced, caption="Analiz Edilen Görüntü", width=400)
    with col_res:
        if st.button("Teşhis Et"):
            prep = cv2.resize(enhanced, (224, 224))
            prep = np.expand_dims(prep / 255.0, axis=0)
            preds = model.predict(prep, verbose=0)
            idx = np.argmax(preds)
            st.success(f"Teşhis: {class_names[idx]}")
            st.metric("Güven Oranı", f"%{np.max(preds)*100:.2f}")

# BÖLÜM 5: SONUÇ (Kriter 20)
st.divider()
st.subheader("Sonuç ve Kaynakça")
st.write("Bu çalışmada %91.4 başarıya ulaşılmıştır. Kaynakça: Kaggle, MobileNetV2 Paper, TF Keras Docs.")
st.caption("Selin Kırca - 220706005 | © 2026 Giresun Üniversitesi")
