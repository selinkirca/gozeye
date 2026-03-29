import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image, ImageOps
import os

# --- 1. AYARLAR VE MODEL YÜKLEME ---
st.set_page_config(page_title="Göz Hastalığı Teşhis Sistemi", layout="centered")

# Kendi dosya yolunu buraya yaz
MODEL_PATH = r'C:\Users\Oguz\Desktop\eye_disease_final_mobilenet_v1.h5'

@st.cache_resource # Modeli her seferinde tekrar yüklememesi için önbelleğe alıyoruz
def load_my_model():
    if os.path.exists(MODEL_PATH):
        return tf.keras.models.load_model(MODEL_PATH)
    else:
        st.error(f"Model dosyası bulunamadı! Lütfen şu yolu kontrol edin: {MODEL_PATH}")
        return None

model = load_my_model()
class_names = ['Cataract', 'Diabetic Retinopathy', 'Glaucoma', 'Normal']

# --- 2. GÖRÜNTÜ ÖN İŞLEME FONKSİYONU (CLAHE) ---
def preprocess_for_model(pil_image):
    # PIL -> OpenCV (RGB)
    img = np.array(pil_image.convert('RGB'))
    
    # CLAHE Uygulama (Kontrast İyileştirme)
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
    lab[:,:,0] = clahe.apply(lab[:,:,0])
    img_enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    
    # Model Boyutu ve Normalizasyon
    img_resized = cv2.resize(img_enhanced, (224, 224))
    img_final = img_resized / 255.0
    return np.expand_dims(img_final, axis=0)

# --- 3. ARAYÜZ TASARIMI ---
st.title("👁️ Akıllı Göz Hastalığı Analiz Sistemi")
st.markdown("""
Bu sistem **MobileNetV2** mimarisi kullanarak retina fotoğraflarını analiz eder. 
*Yüklediğiniz resimler otomatik olarak kontrast iyileştirmesinden geçirilir.*
""")

uploaded_file = st.file_uploader("Bir Retina Fotoğrafı (Fundus) Yükleyin...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None and model is not None:
    # Resmi yükle ve göster
    original_image = Image.open(uploaded_file)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(original_image, caption="Orijinal Resim", use_container_width=True)
    
    with st.spinner('Yapay Zeka Analiz Ediyor...'):
        # --- TTA (Test Time Augmentation) UYGULAMASI ---
        # 1. Orijinal resim tahmini
        img1 = preprocess_for_model(original_image)
        # 2. Yatay çevrilmiş resim tahmini
        img2 = preprocess_for_model(ImageOps.mirror(original_image))
        # 3. Hafif döndürülmüş resim tahmini
        img3 = preprocess_for_model(original_image.rotate(15))
        
        # 3 tahmini de yap ve ortalamasını al
        p1 = model.predict(img1, verbose=0)
        p2 = model.predict(img2, verbose=0)
        p3 = model.predict(img3, verbose=0)
        
        final_preds = (p1 + p2 + p3) / 3.0
        max_idx = np.argmax(final_preds)
        confidence = np.max(final_preds)
        result_label = class_names[max_idx]

    # --- 4. SONUÇLARI GÖSTER ---
    st.divider()
    
    # Güven eşiği kontrolü (%55 altı riskli sayılabilir)
    if confidence < 0.55:
        st.warning(f"⚠️ **Düşük Güven Oranı (%{confidence*100:.1f}):** Model tam emin olamadı. Lütfen daha net bir fotoğraf deneyin.")
    
    st.subheader(f"Tahmini Teşhis: :{ 'green' if result_label == 'Normal' else 'red'}[{result_label}]")
    st.progress(float(confidence))
    st.write(f"Modelin Karar Güveni: **%{confidence*100:.2f}**")

    # Tüm Sınıfların Olasılık Dağılımı
    with st.expander("Detaylı Olasılık Analizi"):
        chart_data = {class_names[i]: float(final_preds[0][i]) for i in range(len(class_names))}
        st.bar_chart(chart_data)

st.info("Not: Bu bir yapay zeka projesidir. Kesin tıbbi sonuçlar için lütfen bir göz doktoruna danışın.")