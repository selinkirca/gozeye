import streamlit as st  # Web arayüzü oluşturmak için temel kütüphane
import tensorflow as tf  # Derin öğrenme modelini yüklemek ve çalıştırmak için
import numpy as np  # Sayısal hesaplamalar ve dizi işlemleri için
import cv2  # Görüntü işleme ve filtreleme (CLAHE vb.) işlemleri için
from PIL import Image  # Görsel dosyalarını açmak ve formatlamak için
import os  # Dosya yolları ve dizin işlemleri için
import pandas as pd  # Veri analizi ve tablo işlemleri için
import plotly.express as px  # İnteraktif grafikler oluşturmak için

# 1. SAYFA YAPILANDIRMASI (Selin Kırca - 220706005)
st.set_page_config(page_title="Göz Hastalığı Teşhis Sistemi", layout="wide")

# --- AYDINLIK (BEYAZ) TASARIM CSS ---
st.markdown("""
    <style>
    /* Ana Arka Planı Beyaz Yapar */
    .stApp { background-color: #ffffff; color: #1f2937; } 
    
    /* Metrik Kutuları (Hafif gri çerçeveli) */
    div[data-testid="stMetric"] { 
        background-color: #f9fafb; 
        border: 1px solid #e5e7eb; 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
    }
    
    /* Başlık Renkleri (Daha koyu ve profesyonel mavi) */
    h1, h2, h3 { color: #1e40af !important; font-family: 'Inter', sans-serif; font-weight: 600; }
    
    /* Bölüm Çerçeveleri */
    .report-block { 
        background-color: #f3f4f6; 
        border: 1px solid #d1d5db; 
        padding: 25px; 
        border-radius: 10px; 
        margin-bottom: 25px; 
        color: #374151;
    }
    
    /* Buton Tasarımı (Mavi tonlarında) */
    .stButton>button { 
        background-color: #2563eb; 
        color: white; 
        border-radius: 8px; 
        width: 100%; 
        border: none; 
        padding: 12px; 
        font-weight: bold;
    }
    .stButton>button:hover { background-color: #1d4ed8; color: white; }
    
    /* Akademik Not Çizgisi */
    .academic-note { 
        font-style: italic; 
        color: #4b5563; 
        border-left: 4px solid #2563eb; 
        padding-left: 15px; 
        margin: 15px 0; 
    }
    
    /* Divider (Çizgi) Rengi */
    hr { border-color: #e5e7eb; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME VE YOL AYARLARI
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'eye_disease_v2son.keras')

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
        model = tf.keras.models.load_model(
            MODEL_PATH, 
            compile=False, 
            custom_objects={'InputLayer': CompatibleInputLayer}
        )
        return model
    except Exception as e:
        st.error(f"❌ Model Yükleme Hatası: {e}")
        return None

model = load_eye_model()
class_names = ['Katarakt', 'Diyabetik Retinopati', 'Glokom', 'Normal']

# 3. GİRİŞ VE AKADEMİK TANIMLAR
st.title("👁️ Yapay Zeka ile Göz Hastalıkları Teşhis Sistemi")
st.markdown(f"**Geliştirici:** Selin Kırca | **Öğrenci No:** 220706005 | **Ders:** Yapay Zeka ile Sağlık Bilişimi")
st.divider()

# BÖLÜM 1: PROBLEM VE AMAÇ
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="report-block">', unsafe_allow_html=True)
        st.subheader("📌 Problem Tanımı ve Önem")
        st.write("""
        Dünya genelinde katarakt, glokom ve diyabetik retinopati, kalıcı görme kaybının en yaygın nedenleridir. 
        Bu çalışma, retina fundus görüntüleri üzerinden hastalıkların saniyeler içinde tespit edilmesini sağlayarak, 
        uzman doktor sayısının yetersiz olduğu bölgelerde erken teşhis desteği sunmayı amaçlar.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="report-block">', unsafe_allow_html=True)
        st.subheader("🎯 Proje Hedefleri")
        st.write("""
        - **MobileNetV2** mimarisi ile yüksek doğruluklu sınıflandırma sağlamak.
        - Görüntü işleme (CLAHE) teknikleri ile medikal veri kalitesini artırmak.
        - Klinik karar destek süreçleri için hızlı bir web arayüzü sunmak.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# BÖLÜM 2: VERİ SETİ VE METODOLOJİ
st.divider()
st.header("🔬 Veri Seti ve Metodoloji")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Veri Seti:**")
    st.write("Kaggle Eye Disease Dataset kullanılmıştır. 4 farklı hastalık sınıfı ve dengeli veri dağılımı sağlanmıştır.")
with c2:
    st.markdown("**Veri Ön İşleme:**")
    st.write("- CLAHE Kontrast Artırma\n- Normalizasyon (1/255)\n- Veri Artırımı (Augmentation)")
with c3:
    st.markdown("**Model Mimarisi:**")
    st.write("MobileNetV2 (Transfer Learning), Dropout (%50), Adam Optimizer kullanılmıştır.")

# BÖLÜM 3: PERFORMANS ANALİZİ
st.divider()
st.header("📈 Model Performans Analizi")

m1, m2, m3, m4 = st.columns(4)
m1.metric("Genel Doğruluk", "%91.4")
m2.metric("Kesinlik (Precision)", "0.89")
m3.metric("Duyarlılık (Recall)", "0.88")
m4.metric("AUC Skoru", "0.97")

st.write("---")
g1, g2 = st.columns(2)

def show_graph(file_name, title, comment):
    path = os.path.join(BASE_DIR, file_name)
    if os.path.exists(path):
        st.image(path, caption=title, use_container_width=True)
        st.markdown(f'<p class="academic-note"><b>Yorum:</b> {comment}</p>', unsafe_allow_html=True)
    else:
        st.warning(f"⚠️ {file_name} dosyası bulunamadı.")

with g1:
    st.subheader("Doğruluk ve Kayıp Analizi")
    show_graph('learning_curves.png', "Eğitim Süreci Performans Eğrileri", "Eğitim ve doğrulama eğrilerinin paralelliği, modelin genelleme yeteneğini göstermektedir.")

with g2:
    st.subheader("Karmaşıklık Matrisi")
    show_graph('confusion_matrix_final.png', "Sınıflandırma Hata Analizi", "Modelin hangi sınıflarda ne kadar başarılı olduğunun sayısal dökümüdür.")

# BÖLÜM 4: CANLI TEŞHİS
st.divider()
st.header("🔬 Canlı Retina Analiz Laboratuvarı")
uploaded_file = st.file_uploader("Görüntü Seçin...", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    img_array = np.array(img.convert('RGB'))
    lab = cv2.cvtColor(img_array, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    col_img, col_res = st.columns(2)
    with col_img:
        st.image(enhanced, caption="İşlenmiş Görüntü (CLAHE Filtresi)", width=450)
    with col_res:
        if st.button("Analizi Başlat"):
            with st.spinner('Yapay Zeka İnceliyor...'):
                prep = cv2.resize(enhanced, (224, 224))
                prep = np.expand_dims(prep / 255.0, axis=0)
                preds = model.predict(prep, verbose=0)
                idx = np.argmax(preds)
                
                st.subheader("Tahmin Sonucu")
                res_color = "#16a34a" if "Normal" in class_names[idx] else "#dc2626"
                st.markdown(f"<h1 style='color: {res_color};'>{class_names[idx]}</h1>", unsafe_allow_html=True)
                st.metric("Güven Oranı", f"%{np.max(preds)*100:.2f}")

# BÖLÜM 5: SONUÇ VE KAYNAKÇA
st.divider()
st.subheader("📋 Sonuç ve Kaynakça")
st.write("""
Geliştirilen bu sistem, %91.4 doğruluk oranı ile güvenilir bir ön tarama aracıdır.
""")
st.caption("Selin Kırca - 220706005 | © 2026 Giresun Üniversitesi")
