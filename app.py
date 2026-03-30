import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px

# 1. SAYFA YAPILANDIRMASI (Selin Kırca - 220706005)
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

# 2. MODEL YÜKLEME
MODEL_PATH = 'eye_disease_v2son.keras'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error("Model dosyası sunucuda bulunamadı.")
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
    except Exception: return None

model = load_eye_model()
class_names = ['Katarakt', 'Diyabetik Retinopati', 'Glokom', 'Normal']

# 3. GİRİŞ BİLGİLERİ
st.title("👁️ Derin Öğrenme ile Göz Hastalıkları Teşhis Sistemi")
st.markdown(f"**Geliştirici:** Selin Kırca | **Öğrenci No:** 220706005 | **Üniversite:** Giresun Üniversitesi")
st.divider()

# BÖLÜM 1: PROBLEM VE AMAÇ
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="report-block">', unsafe_allow_html=True)
        st.subheader("Problem Tanımı ve Çalışmanın Önemi")
        st.write("""
        Küresel ölçekte katarakt, glokom ve diyabetik retinopati, kalıcı görme kaybının en yaygın nedenleridir. 
        Bu çalışma, retina fundus görüntüleri üzerinden hastalıkların saniyeler içinde tespit edilmesini sağlayarak, 
        erken teşhis süreçlerine dijital bir destek sunmayı amaçlamaktadır.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="report-block">', unsafe_allow_html=True)
        st.subheader("Proje Hedefleri")
        st.write("""
        - MobileNetV2 mimarisi ile yüksek doğruluklu sınıflandırma sağlamak.
        - Görüntü işleme teknikleri ile medikal veri kalitesini artırmak.
        - Kullanıcı dostu bir klinik karar destek arayüzü oluşturmak.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# BÖLÜM 2: VERİ SETİ VE METODOLOJİ
st.divider()
st.header("Veri Seti ve Uygulanan Metodoloji")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Veri Seti:**")
    st.write("Kaggle Eye Disease Dataset. 4 farklı hastalık sınıfı ve dengeli veri dağılımı.")
with c2:
    st.markdown("**Ön İşleme:**")
    st.write("- CLAHE Filtresi\n- Normalizasyon\n- Veri Artırımı (Augmentation)")
with c3:
    st.markdown("**Eğitim Ayrımı:**")
    st.write("%80 Eğitim, %20 Doğrulama verisi kullanılmıştır.")

# BÖLÜM 3: PERFORMANS ANALİZİ
st.divider()
st.header("Performans Metrikleri ve Grafiksel Analiz")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Genel Doğruluk", "%91.4")
m2.metric("Kesinlik", "0.89")
m3.metric("Duyarlılık", "0.88")
m4.metric("AUC Skoru", "0.97")

g1, g2 = st.columns(2)
with g1:
    st.subheader("Eğitim Süreci Analizi")
    # Dosya yolunu 'learning_curves.png' olarak güncelledik
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Doğruluk ve Kayıp Grafikleri", width=600)
    st.markdown('<p class="academic-note">Yorum: Eğitim ve doğrulama eğrilerinin paralelliği, modelin aşırı öğrenme (overfitting) yapmadan genelleme yeteneği kazandığını kanıtlamaktadır.</p>', unsafe_allow_html=True)

with g2:
    st.subheader("Karmaşıklık Matrisi (Confusion Matrix)")
    # Dosya yolunu 'confusion_matrix_final.png' olarak güncelledik
    if os.path.exists('confusion_matrix_final.png'):
        st.image('confusion_matrix_final.png', caption="Sınıflandırma Hata Analizi", width=600)
    st.markdown('<p class="academic-note">Yorum: Model Normal ve DR sınıflarında yüksek başarı gösterirken, Glokom ve Normal arasındaki benzerlikler kısıtlı karışıklığa yol açmıştır.</p>', unsafe_allow_html=True)

st.subheader("ROC Eğrisi Analizi")
# Dosya yolunu 'roc_curve_final.png' olarak güncelledik
if os.path.exists('roc_curve_final.png'):
    st.image('roc_curve_final.png', caption="Sınıf Bazlı AUC Analizi", width=700)
st.markdown('<p class="academic-note">Yorum: AUC değerlerinin 0.95 üzerinde olması, modelin sağlıklı ve hastalıklı dokuyu ayırt etme gücünün mükemmele yakın olduğunu gösterir.</p>', unsafe_allow_html=True)

# BÖLÜM 4: CANLI TEŞHİS
st.divider()
st.header("🔬 Retina Analiz Laboratuvarı")
uploaded_file = st.file_uploader("Analiz için görüntü yükleyiniz...", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    col_img, col_res = st.columns(2)
    with col_img:
        st.image(enhanced, caption="İşlenmiş Görüntü", width=400)
    with col_res:
        if st.button("Teşhis Et"):
            prep = cv2.resize(enhanced, (224, 224))
            prep = np.expand_dims(prep / 255.0, axis=0)
            preds = model.predict(prep, verbose=0)
            idx = np.argmax(preds)
            st.success(f"Tahmin Edilen Sonuç: {class_names[idx]}")
            st.metric("Güven Oranı", f"%{np.max(preds)*100:.2f}")

# BÖLÜM 5: SONUÇ VE KAYNAKÇA
st.divider()
st.subheader("Sonuç ve Kaynakça")
st.write("""
Derin öğrenme teknikleri ile geliştirilen bu sistem, retina hastalıklarının teşhisinde %91.4 doğruluk başarısına ulaşmıştır. 
\n**Kaynakça:**
- Sandler, M., et al. (2018). MobileNetV2.
- Kaggle: Eye Disease Dataset.
- TensorFlow Keras API.
""")
st.caption("Selin Kırca - 220706005 | © 2026 Giresun Üniversitesi")
