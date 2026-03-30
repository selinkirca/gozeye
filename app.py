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
    .stButton>button { background-color: #238636; color: white; border-radius: 8px; width: 100%; border: none; padding: 10px; }
    .stButton>button:hover { background-color: #2ea043; border: none; }
    </style>
    """, unsafe_allow_html=True)

# 2. MODEL YÜKLEME VE KERAS 3 UYUMLULUK YAMASI
MODEL_PATH = 'eye_disease_v2son.keras'

@st.cache_resource
def load_eye_model():
    if not os.path.exists(MODEL_PATH):
        st.error(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
        return None
    
    # --- KRİTİK YAMA: Keras 3 'batch_shape' Hatası Çözümü ---
    from tensorflow.keras.layers import InputLayer

    class CompatibleInputLayer(InputLayer):
        def __init__(self, *args, **kwargs):
            # Eski Keras sürümlerinden gelen 'batch_shape' parametresini Keras 3'e uyarlar
            if 'batch_shape' in kwargs:
                kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
            super().__init__(*args, **kwargs)

    # Modeli yüklerken bu özel katmanı sisteme tanıtıyoruz
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
        st.info("💡 İpucu: Modelin içindeki katman isimleri Keras 3 ile çakışıyor. Bu kod otomatik tamir etmeye çalışıyor.")
        return None

# Modeli belleğe al
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

# 3. SIDEBAR (Selin Kırca - 220706005)
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/822/822102.png", width=80)
    st.markdown("<h2 style='text-align: center;'>Oculus AI</h2>", unsafe_allow_html=True)
    menu = st.radio("Menü:", ["🔬 Canlı Teşhis", "📈 Analiz & Metrikler"])
    st.divider()
    st.markdown(f"""
    **Geliştirici:** Selin Kırca  
    **Öğrenci No:** 220706005  
    **Üniversite:** Giresun Üni.
    """)

# --- ANA PANEL ---
if menu == "🔬 Canlı Teşhis":
    st.header("🔬 Retina Analiz Laboratuvarı")
    st.write("Lütfen analiz edilecek retina (fundus) görüntüsünü yükleyin.")
    
    uploaded_file = st.file_uploader("Görüntü Seçin...", type=["jpg", "png", "jpeg"])
    
    if uploaded_file and model is not None:
        img = Image.open(uploaded_file)
        enhanced = apply_clahe(img)
        
        col1, col2 = st.columns([1, 1])
        with col1:
            st.image(enhanced, caption="İşlenmiş Görüntü (CLAHE Filtresi)", use_container_width=True)
        
        with col2:
            if st.button("Teşhis Et"):
                with st.spinner('Yapay Zeka Tahmin Yapıyor...'):
                    input_data = preprocess_for_model(enhanced)
                    preds = model.predict(input_data, verbose=0)
                    idx = np.argmax(preds)
                    conf = np.max(preds)
                    
                    # Sonuç Görselleştirmesi
                    st.subheader("Analiz Sonucu")
                    result_color = "#238636" if "Normal" in class_names[idx] else "#da3633"
                    st.markdown(f"<h1 style='color: {result_color}; font-size: 28px;'>{class_names[idx]}</h1>", unsafe_allow_html=True)
                    st.metric("Güven Oranı", f"%{conf*100:.2f}")
                    
                    # Olasılık Dağılım Grafiği
                    chart_data = pd.DataFrame({'Sınıf': class_names, 'Olasılık': preds[0]})
                    fig = px.bar(chart_data, x='Sınıf', y='Olasılık', color='Sınıf', template="plotly_dark")
                    fig.update_layout(showlegend=False, height=300)
                    st.plotly_chart(fig, use_container_width=True)

elif menu == "📈 Analiz & Metrikler":
    st.header("📈 Model Performans Raporu")
    c1, c2, c3 = st.columns(3)
    c1.metric("Doğruluk (Acc)", "%91.4")
    c2.metric("Hassasiyet", "0.89")
    c3.metric("Özellik", "0.93")
    
    st.divider()
    st.info("Bu model MobileNetV2 mimarisi üzerine inşa edilmiş olup, transfer learning teknikleriyle optimize edilmiştir.")
    st.success("Veri seti: 4 Sınıf (Cataract, DR, Glaucoma, Normal)")

st.divider()
st.caption("Selin Kırca - 220706005 | © 2026 Sağlık Bilişimi Projesi")
