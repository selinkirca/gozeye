import os  # İşletim sistemi dosya işlemleri için
import numpy as np  # Matematiksel ve dizi operasyonları için
import matplotlib.pyplot as plt  # Grafik çizimleri için ana kütüphane
import seaborn as sns  # Isı haritaları (Heatmap) gibi gelişmiş görseller için
import tensorflow as tf  # Model yükleme ve çalıştırma için
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Test verilerini hazırlamak için
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc  # İstatistiksel metrikler
from sklearn.preprocessing import label_binarize  # ROC analizi için etiketleri 0-1 formatına dönüştürmek için

# ==========================================
# 1. AYARLAR VE YAPILANDIRMA (Selin Kırca - 220706005)
# ==========================================
DATASET_PATH = r'C:\Users\selin\OneDrive\Masaüstü\dataset' 
MODEL_PATH = 'eye_disease_v2son.keras'
IMG_SIZE = (224, 224) # Modelin eğitildiği giriş boyutu
BATCH_SIZE = 32

# ==========================================
# 2. DOĞRULAMA VERİ YÜKLEYİCİ (Data Loader)
# ==========================================
# Test verilerini modele sunmadan önce normalize eder ve %20'lik dilimi ayırır.
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='categorical', # Çok sınıflı sınıflandırma
    subset='validation',      # Sadece doğrulama verilerini al
    shuffle=False             # Tahminlerin sırasının bozulmaması için False olmalı
)

# ==========================================
# 3. MODELİN YÜKLENMESİ VE UYUMLULUK YAMASI
# ==========================================
# Keras sürümleri arasındaki girdi katmanı isimlendirme farklarını çözen yardımcı sınıf.
from tensorflow.keras.layers import InputLayer
class CompatibleInputLayer(InputLayer):
    def __init__(self, *args, **kwargs):
        if 'batch_shape' in kwargs:
            kwargs['batch_input_shape'] = kwargs.pop('batch_shape')
        super().__init__(*args, **kwargs)

print("🚀 Model yükleniyor...")
model = tf.keras.models.load_model(
    MODEL_PATH, 
    compile=False, 
    custom_objects={'InputLayer': CompatibleInputLayer}
)

# ==========================================
# 4. TAHMİN OPERASYONU (Inference)
# ==========================================
print("🔍 Tahminler yapılıyor...")
val_gen.reset() # Veri akışını başa sar
preds = model.predict(val_gen) # Modelden tüm test verisi için olasılıkları al
y_pred = np.argmax(preds, axis=1) # En yüksek olasılıklı sınıfın indeksini al (0, 1, 2, 3)
y_true = val_gen.classes # Gerçek hastalık etiketlerini al
labels = list(val_gen.class_indices.keys()) # Sınıf isimlerini (Katarakt vb.) al

# ==========================================
# 5. PERFORMANS GÖRSELLEŞTİRME FONKSİYONLARI
# ==========================================

# --- METRİK 1: KARMAŞIKLIK MATRİSİ (Confusion Matrix) ---
# Amacı: Modelin hangi hastalığı hangisiyle karıştırdığını sayısal olarak görmek.
def save_confusion_matrix(y_true, y_pred, labels):
    cm = confusion_matrix(y_true, y_pred) # Hata matrisini hesapla
    plt.figure(figsize=(10, 8))
    # Heatmap (Isı Haritası) ile görselleştir
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.title('Oculus AI: Karmaşıklık Matrisi (v2)')
    plt.ylabel('Gerçek Etiket')
    plt.xlabel('Tahmin Edilen Etiket')
    plt.savefig('confusion_matrix_v2.png', dpi=300, bbox_inches='tight')
    plt.close() # Belleği temizle
    print("✅ confusion_matrix_v2.png kaydedildi.")

# --- METRİK 2: ROC EĞRİSİ ANALİZİ ---
# Amacı: Modelin sınıfları birbirinden ayırma gücünü ölçer. AUC=1.0 mükemmeldir.
def save_roc_curve(y_true, preds, labels):
    n_classes = len(labels)
    # Etiketleri ROC hesaplaması için ikili sisteme (One-vs-Rest) çevirir
    y_true_bin = label_binarize(y_true, classes=range(n_classes))
    
    plt.figure(figsize=(10, 8))
    colors = ['blue', 'red', 'green', 'orange']
    
    # Her bir sınıf (hastalık) için ayrı bir eğri çizdir
    for i, color in zip(range(n_classes), colors):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], preds[:, i])
        roc_auc = auc(fpr, tpr) # Eğri altında kalan alanı hesapla
        plt.plot(fpr, tpr, color=color, lw=2, label=f'{labels[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--', lw=2) # Şans eseri tahmin çizgisi (diagonal)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (Yanlış Pozitif Oranı)')
    plt.ylabel('True Positive Rate (Doğru Pozitif Oranı)')
    plt.title('Oculus AI: ROC Eğrisi Analizi (v2)')
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.savefig('roc_curve_v2.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ roc_curve_v2.png kaydedildi.")

# --- METRİK 3: EĞİTİM ANALİZİ (Learning Curves) ---
# Amacı: Eğitim sırasında doğruluğun nasıl arttığını ve hatanın (loss) nasıl düştüğünü izlemek.
def save_learning_curves(history):
    plt.figure(figsize=(12, 5))
    
    # Başarı (Accuracy) Grafiği
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Eğitim Başarısı')
    plt.plot(history.history['val_accuracy'], label='Doğrulama Başarısı')
    plt.title('Model Doğruluğu (Accuracy)')
    plt.xlabel('Epoch')
    plt.ylabel('Doğruluk')
    plt.legend()
    
    # Kayıp (Loss) Grafiği
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Eğitim Kaybı')
    plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
    plt.title('Model Kaybı (Loss)')
    plt.xlabel('Epoch')
    plt.ylabel('Kayıp')
    plt.legend()
    
    plt.savefig('learning_curves.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ learning_curves.png kaydedildi.")

# ==========================================
# 6. ANALİZİ BAŞLAT VE RAPORLA
# ==========================================
# Grafikleri dosyaya kaydet
save_confusion_matrix(y_true, y_pred, labels)
save_roc_curve(y_true, preds, labels)

# Konsola detaylı Precision, Recall ve F1-Skor raporu bas
print("\n📑 Sınıflandırma Raporu:")
print(classification_report(y_true, y_pred, target_names=labels))