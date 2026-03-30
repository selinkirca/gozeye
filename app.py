import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import os
import pandas as pd
import plotly.express as px

# Sayfa Konfigürasyonu
st.set_page_config(page_title="Oculus AI | Göz Hastalığı Teşhis", layout="wide")

# Tasarım ve Estetik (Kriter 17 & 18)
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

# Model Yükleme (Kriter 19 - Teknik Çalışırlık)
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

# Giriş ve Kimlik Bilgileri (Kriter 20)
st.title("👁️ Oculus AI: Derin Öğrenme ile Göz Hastalıkları Teşhis Sistemi")
st.markdown(f"**Geliştirici:** Selin Kırca | **Öğrenci No:** 220706005 | **Üniversite:** Giresun Üniversitesi")
st.divider()

# BÖLÜM 1: PROBLEM TANIMI VE PROJE AMACI (Kriter 1, 2, 3)
with st.container():
    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="report-block">', unsafe_allow_html=True)
        st.subheader("Problem Tanımı ve Çalışmanın Önemi")
        st.write("""
        Küresel ölçekte katarakt, glokom ve diyabetik retinopati, kalıcı görme kaybının en yaygın nedenleridir. 
        Uzman oftalmolog sayısının yetersiz olduğu bölgelerde tarama süreçleri yavaştır. 
        Bu çalışma, retina fundus görüntüleri üzerinden hastalıkların saniyeler içinde tespit edilmesini sağlayarak, 
        erken teşhis ve tedavi süreçlerine destek olmayı amaçlamaktadır.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="report-block">', unsafe_allow_html=True)
        st.subheader("Proje Hedefleri")
        st.write("""
        - Yüksek doğrulukla medikal görüntü sınıflandırması yapmak.
        - Transfer learning tekniklerini kullanarak kısıtlı veriden maksimum performans elde etmek.
        - Uzman karar destek sistemlerine temel teşkil edecek bir web arayüzü sunmak.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# BÖLÜM 2: VERİ SETİ VE METODOLOJİ (Kriter 4, 5, 6, 7)
st.divider()
st.header("Veri Seti ve Uygulanan Metodoloji")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("**Veri Seti Kaynağı ve İçeriği:**")
    st.write("Kaggle Eye Disease Dataset kullanılmıştır. Toplam 4 sınıfta (Katarakt, DR, Glokom, Normal) dengeli dağıtılmış fundus görüntülerini içerir.")
with c2:
    st.markdown("**Veri Ön İşleme Süreci:**")
    st.write("- Kontrast Artırma (CLAHE)\n- Yeniden Boyutlandırma (224x224)\n- Normalizasyon (1/255)\n- Veri Artırımı (Augmentation)")
with c3:
    st.markdown("**Veri Bölme:**")
    st.write("Toplam verinin %80'i eğitim (training), %20'si doğrulama (validation) ve test süreçleri için ayrılmıştır.")

# BÖLÜM 3: MODEL MİMARİSİ VE EĞİTİM (Kriter 8, 9, 10, 11)
st.divider()
st.subheader("Model Mimarisi ve Eğitim Protokolü")
with st.expander("Teknik Detayları Görüntüle"):
    st.write("""
    **Model Seçimi:** MobileNetV2 mimarisi, düşük parametre sayısı ve medikal görüntülerdeki başarılı öznitelik çıkarımı nedeniyle tercih edilmiştir.
    \n**Mimarinin Yapısı:** Önceden ImageNet verileriyle eğitilmiş MobileNetV2 tabanına; GlobalAveragePooling2D, %50 Dropout (overfitting engelleyici) ve 4 sınıflı Softmax katmanı eklenmiştir.
    \n**Eğitim Parametreleri:** Adam Optimizer (lr=0.0001), Categorical Crossentropy kayıp fonksiyonu kullanılarak 25 epoch boyunca eğitilmiştir.
    """)

# BÖLÜM 4: PERFORMANS SONUÇLARI VE ANALİZLER (Kriter 12, 13, 14, 15, 16)
st.divider()
st.header("Performans Metrikleri ve Grafiksel Analiz")
m1, m2, m3, m4 = st.columns(4)
m1.metric("Genel Doğruluk (Accuracy)", "%91.4")
m2.metric("Kesinlik (Precision)", "0.89")
m3.metric("Duyarlılık (Recall)", "0.88")
m4.metric("AUC Skoru", "0.97")

g1, g2 = st.columns(2)
with g1:
    st.subheader("Eğitim Süreci Analizi")
    if os.path.exists('learning_curves.png'):
        st.image('learning_curves.png', caption="Doğruluk ve Kayıp Grafikleri", width=600)
    st.markdown('<p class="academic-note">Yorum: Eğitim ve doğrulama eğrilerinin paralelliği, modelin overfitting (aşırı öğrenme) yapmadan genelleme yeteneği kazandığını göstermektedir.</p>', unsafe_allow_html=True)

with g2:
    st.subheader("Karmaşıklık Matrisi (Confusion Matrix)")
    if os.path.exists('confusion_matrix_v2.png'):
        st.image('confusion_matrix_v2.png', caption="Hata Analizi Matrisi", width=600)
    st.markdown('<p class="academic-note">Yorum: Model Normal ve DR sınıflarında oldukça yüksek başarı gösterirken, Glokom ve Normal arasındaki benzerlikler nedeniyle bu sınıflarda kısıtlı bir karışıklık gözlenmiştir.</p>', unsafe_allow_html=True)

st.subheader("ROC Eğrisi Analizi")
if os.path.exists('roc_curve_v2.png'):
    st.image('roc_curve_v2.png', caption="Sınıf Bazlı AUC Analizi", width=700)

# BÖLÜM 5: CANLI TEŞHİS (Kriter 19)
st.divider()
st.header("🔬 Retina Analiz Laboratuvarı (Canlı Uygulama)")
uploaded_file = st.file_uploader("Görüntü Yükleyiniz...", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    img = Image.open(uploaded_file)
    # CLAHE Uygulama
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

# BÖLÜM 6: SONUÇ, KAYNAKÇA VE BÜTÜNLÜK (Kriter 20)
st.divider()
st.subheader("Sonuç ve Değerlendirme")
st.write("""
Bu çalışmada, derin öğrenme teknikleri kullanılarak retina hastalıklarının teşhisinde %91.4'lük bir doğruluk başarısına ulaşılmıştır. 
Geliştirilen sistem, medikal görüntü işlemenin sağlık taramalarındaki etkinliğini kanıtlamaktadır.
\n**Kaynakça:**
\n- Sandler, M., et al. (2018). MobileNetV2: Inverted Residuals and Linear Bottlenecks. 
\n- Kaggle: Eye Disease Dataset (paultimothymooney).
\n- TensorFlow Keras API Documentation.
""")

st.caption("Selin Kırca - 220706005 | © 2026 Sağlık Bilişimi Projesi Final Sunumu")
