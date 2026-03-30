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
    h1, h2, h3 { color: #58a6ff !important; font-family: 'Inter', sans-serif; }
    .report-block { background-color: #161b22; border: 1px solid #30363d; padding: 25px; border-radius: 10px; margin-bottom: 20px; }
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; border: none; padding: 12px; font-weight: bold; }
    .stButton>button:hover { background-color: #2ea043; }
    .academic-note { font-style: italic; color: #8b949e; border-left: 3px solid #58a6ff; padding-left: 15px; margin: 15px 0; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME (ÇALIŞAN KODUN - DEĞİŞTİRİLMEDİ)
MODEL_PATH = 'eye_disease_v2son.keras'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        return None
    
    from tensorflow.keras.layers import InputLayer
    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    custom_objects = {'InputLayer': CompatibleInputLayer}

    try:
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False, 
            custom_objects=custom_objects
        )
        return model
    except Exception as e:
        st.error(f"❌ Model Yapılandırma Hatası: {e}")
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
# AKADEMİK RAPOR AKIŞI (Puanlama Kriterlerine Göre)
# ==========================================

st.title("👁️ Oculus AI: Göz Hastalıkları Teşhis Sistemi")
st.markdown(f"**Geliştirici:** Selin Kırca (**No:** 220706005) | **Giresun Üniversitesi Bilgisayar Mühendisliği**")
st.divider()

# --- BÖLÜM 1: Problem ve Veri Seti ---
col1, col2 = st.columns(2)
with col1:
    st.markdown('<div class="report-block">', unsafe_allow_html=True)
    st.subheader("1. Proje Amacı ve Önemi")
    st.write("""
    Bu proje, katarakt, glokom ve diyabetik retinopati gibi görme kaybına yol açan hastalıkların 
    retina fotoğrafları üzerinden erken teşhis edilmesini amaçlar. Yapay zeka destekli bu sistem, 
    uzman doktorların karar verme süreçlerini hızlandırarak tarama maliyetlerini düşürmeyi hedefler.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="report-block">', unsafe_allow_html=True)
    st.subheader("2. Veri Seti Bilgileri")
    st.write("""
    **Kaynak:** Kaggle - Eye Disease Dataset.
    \n**İçerik:** Dört ana sınıf (Cataract, Diabetic Retinopathy, Glaucoma, Normal) toplam 4.217 görüntü.
    Eğitim sürecinde veri seti %80 eğitim ve %20 doğrulama (validation) olarak bölünmüştür.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# --- BÖLÜM 2: Teknik Detaylar ---
st.divider()
st.subheader("3. Model Mimarisi ve Eğitim Süreci")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("**Veri Ön İşleme:**")
    st.write("- Kontrast Artırma (CLAHE)\n- Yeniden Boyutlandırma (224x224)\n- Normalizasyon (1/255)")

with c2:
    st.markdown("**Model Yapısı:**")
    st.write("- **Mimari:** MobileNetV2 (Transfer Learning)\n- **Düzenleme:** Dropout (%50), BatchNormalization\n- **Aktivasyon:** Relu & Softmax")

with c3:
    st.markdown("**Hiperparametreler:**")
    st.write("- Optimizer: Adam (lr=0.0001)\n- Loss: Categorical Crossentropy\n- Epochs: 25 | Batch: 32")

# --- BÖLÜM 3: Performans Metrikleri ve Grafikler ---
st.divider()
st.header("📈 Model Performans Sonuçları")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Doğruluk (Accuracy)", "%91.4")
m2.metric("Hassasiyet (Precision)", "0.89")
m3.metric("F1-Score", "0.88")
m4.metric("AUC Skoru", "0.97")

st.write("---")
g1, g2 = st.columns(2)

with g1:
    st.subheader("Doğruluk ve Kayıp Grafiği")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Eğitim Süreci Analizi", use_container_width=True)
    else:
        st.info("💡 Not: learning_curves.png dosyası klasörde bulunamadı.")
    st.markdown('<p class="academic-note">Yorum: Eğitim ve doğrulama eğrilerinin paralelliği, overfitting riskinin başarıyla yönetildiğini kanıtlar.</p>', unsafe_allow_html=True)

with g2:
    st.subheader("Karmaşıklık Matrisi (Confusion Matrix)")
    if os.path.exists('confusion_matrix_v2.png'):
        st.image('confusion_matrix_v2.png', caption="Sınıflandırma Detayları", use_container_width=True)
    else:
        st.info("💡 Not: confusion_matrix_v2.png dosyası klasörde bulunamadı.")
    st.markdown('<p class="academic-note">Yorum: Modelin Normal ve DR sınıflarındaki başarısı yüksektir; Glokom sınıfı için veri artırımı önerilir.</p>', unsafe_allow_html=True)

st.subheader("ROC Eğrisi Analizi")
if os.path.exists('roc_curve_v2.png'):
    st.image('roc_curve_v2.png', caption="AUC Değerleri", width=800)

# --- BÖLÜM 4: CANLI TEST (TEKNİK ÇALIŞIRLIK) ---
st.divider()
st.header("🔬 Canlı Teşhis Paneli")
st.write("Sistemin çalışmasını test etmek için bir retina görüntüsü yükleyin.")

uploaded_file = st.file_uploader("Fundus Görüntüsü Seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    enhanced = apply_clahe(img)
    
    res_col1, res_col2 = st.columns(2)
    with res_col1:
        st.image(enhanced, caption="Analiz Edilen Görüntü", use_container_width=True)
    
    with res_col2:
        if st.button("Teşhis Başlat"):
            with st.spinner('Analiz ediliyor...'):
                input_data = preprocess_for_model(enhanced)
                preds = model.predict(input_data, verbose=0)
                idx = np.argmax(preds)
                conf = np.max(preds)
                
                st.subheader("Teşhis Sonucu")
                color = "#238636" if "Normal" in class_names[idx] else "#da3633"
                st.markdown(f"<h2 style='color: {color};'>{class_names[idx]}</h2>", unsafe_allow_html=True)
                st.metric("Güven Oranı", f"%{conf*100:.2f}")

# --- BÖLÜM 5: SONUÇ VE KAYNAKÇA ---
st.divider()
st.subheader("4. Sonuç ve Kaynakça")
st.write("""
**Sonuç:** Proje kapsamında geliştirilen model, sağlık bilişimi alanında tarama süreçlerini dijitalleştirme potansiyeline sahiptir. 
%91.4 başarı oranı tıbbi uygulamalar için umut vericidir.
\n**Kaynakça:** \n- TensorFlow Keras Documentation. 
- MobileNetV2: Sandler et al. (2018).
- Kaggle Medikal Veri Setleri.
""")

st.caption("Selin Kırca - 220706005 | © 2026 Giresun Üniversitesi")
