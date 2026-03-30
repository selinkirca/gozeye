import os  # Dosya ve dizin yollarını yönetmek için
import numpy as np  # Sayısal veri manipülasyonu için
import tensorflow as tf  # Derin öğrenme motoru
from tensorflow.keras import layers, models, callbacks, regularizers  # Model katmanları ve araçları
from tensorflow.keras.preprocessing.image import ImageDataGenerator  # Görüntüleri zenginleştirme (Augmentation)
from sklearn.utils import class_weight  # Sınıf dengesizliğini gidermek için (Opsiyonel kullanım için)

# ==========================================
# 1. KONFİGÜRASYON (Selin Kırca - 220706005)
# ==========================================
# Veri setinin bilgisayardaki tam yolu ve eğitim parametreleri
DATASET_PATH = r'C:\Users\selin\OneDrive\Masaüstü\dataset' 
IMG_SIZE = (224, 224)  # MobileNetV2'nin standart giriş boyutu
BATCH_SIZE = 32        # Her adımda modele beslenecek görüntü sayısı
EPOCHS = 30            # Tüm veri setinin modelden geçme sayısı

# ==========================================
# 2. GÜÇLENDİRİLMİŞ DATA AUGMENTATION (Veri Artırımı)
# ==========================================
# Amacı: Elimizdeki sınırlı resmi döndürerek, kaydırarak veya çevirerek yapay yolla çoğaltmak.
# Bu sayede model nesneyi sadece dik değil, her açıdan tanımayı öğrenir (Overfitting Engelleyici).
train_datagen = ImageDataGenerator(
    rescale=1./255,          # Piksel değerlerini 0-255'ten 0-1 arasına normalize eder
    rotation_range=40,       # Resimleri rastgele 40 dereceye kadar döndürür
    width_shift_range=0.3,    # Resmi yatayda %30 oranında sağa-sola kaydırır
    height_shift_range=0.3,   # Resmi dikeyde %30 oranında yukarı-aşağı kaydırır
    shear_range=0.2,          # Resme "eğme" deformasyonu uygular
    zoom_range=0.3,           # Rastgele yakınlaştırma veya uzaklaştırma yapar
    horizontal_flip=True,     # Resmi yatayda aynalar (sol göz -> sağ göz gibi)
    vertical_flip=True,       # Resmi dikeyde aynalar (retina görüntülerinde yön fark etmez)
    fill_mode='nearest',      # Kaydırma sonrası oluşan boşlukları en yakın piksellerle doldurur
    validation_split=0.2      # Mevcut verinin %20'sini test/doğrulama için ayırır
)

# Eğitim verisi akışı (Generator)
train_gen = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='training', shuffle=True
)

# Doğrulama (Validation) verisi akışı
val_gen = train_datagen.flow_from_directory(
    DATASET_PATH, target_size=IMG_SIZE, batch_size=BATCH_SIZE,
    class_mode='categorical', subset='validation', shuffle=False
)

# ==========================================
# 3. MOBILENETV2 + TRANSFER LEARNING (Önceden Eğitilmiş Model)
# ==========================================
# ImageNet üzerinde eğitilmiş devasa bir bilgi birikimini (MobileNetV2) temel alıyoruz.
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(224, 224, 3), include_top=False, weights='imagenet'
)
# Temel modelin (MobileNet) ağırlıklarını donduruyoruz; sadece bizim eklediğimiz katmanlar eğitilecek.
base_model.trainable = False 

# Kendi mimarimizi inşa ediyoruz
model = models.Sequential([
    base_model,  # Temel özellik çıkarıcı katman
    layers.GlobalAveragePooling2D(),  # 2B veriyi tek boyuta (vektöre) indirger
    
    # Yeni Karar Katmanları ve Düzenlileştirme (Regularization)
    # L2: Ağırlıkların aşırı büyümesini cezalandırarak modelin karmaşıklığını sınırlar.
    layers.Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(), # Eğitim sırasında verileri standartlaştırarak hızı ve kararlılığı artırır
    layers.Dropout(0.5),         # Nöronların %50'sini rastgele "öldürerek" modelin belirli nöronlara bağımlı kalmasını önler
    
    layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(0.01)),
    layers.BatchNormalization(),
    layers.Dropout(0.4),         # İkinci bir Dropout ile güvenliği artırıyoruz
    
    # Çıkış Katmanı: Sınıf sayısı kadar nöron ve olasılık dağılımı için Softmax
    layers.Dense(train_gen.num_classes, activation='softmax')
])

# ==========================================
# 4. DERLEME (Modelin Çalışma Kuralları)
# ==========================================
# Düşük bir öğrenme hızı (0.00005) seçiyoruz; model "koşmak" yerine "dikkatlice adım atmalı".
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.00005),
    loss='categorical_crossentropy', # Çok sınıflı sınıflandırma kaybı fonksiyonu
    metrics=['accuracy']             # Takip edilecek ana metrik: Doğruluk
)

# ==========================================
# 5. AKILLI DENETLEYİCİLER (Callbacks)
# ==========================================
# EarlyStopping: Eğer doğrulama kaybı (val_loss) 8 dönem boyunca iyileşmezse eğitimi durdurur.
# restore_best_weights: Durduğunda modelin en başarılı olduğu andaki ağırlıkları geri yükler.
early_stop = callbacks.EarlyStopping(
    monitor='val_loss', patience=8, restore_best_weights=True
)

# ReduceLROnPlateau: Eğitim tıkandığında (4 dönem boyunca iyileşme yoksa) öğrenme hızını %20'sine düşürür.
# Bu, modelin minimum hata noktasına daha hassas yaklaşmasını sağlar (Vites küçültme).
reduce_lr = callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=4, min_lr=0.000001
)

# ==========================================
# 6. EĞİTİM SÜRECİ
# ==========================================
print("\n--- Oculus AI: Gelişmiş MobileNet Eğitimi Başlıyor ---")
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=val_gen,
    callbacks=[early_stop, reduce_lr] # Tanımladığımız akıllı araçları buraya bağlıyoruz
)

# ==========================================
# 7. KAYDETME
# ==========================================
# Modeli yeni nesil .keras formatında kaydediyoruz (Yapı + Ağırlıklar + Optimizasyon durumu).
model.save('oculus_ai_mobilenet_v3.keras')
print("\n✅ Yeni nesil model kaydedildi: oculus_ai_mobilenet_v3.keras")