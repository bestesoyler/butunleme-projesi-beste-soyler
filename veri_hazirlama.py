import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

# 1. Veriyi Yükleme
# İndirdiğiniz dataset.csv dosyasının kod ile aynı klasörde olduğundan emin olun.
try:
    df = pd.read_csv("dataset.csv", sep=';') # Veri seti noktalı virgül ile ayrılmış olabilir, duruma göre ',' yapabilirsiniz.
    print("Veri seti başarıyla yüklendi!")
except FileNotFoundError:
    print("Hata: dataset.csv dosyası bulunamadı.")

# 2. Veri Temizleme ve Ön İşleme
# Eksik verileri düşürme
df = df.dropna()

# Hedef değişkeni belirleme (Kaggle'daki veri setinde genellikle 'Target' sütunudur)
X = df.drop('Target', axis=1)
y = df['Target']

# Kategorik hedef değişkenini (Dropout, Enrolled, Graduate) sayısal değerlere dönüştürme
le = LabelEncoder()
y = le.fit_transform(y)

# 3. Eğitim ve Test Setlerine Bölme
# Verinin %80'i eğitim, %20'si test için ayrılıyor.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Veri Ölçeklendirme (Standartlaştırma)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("Veri temizleme ve hazırlık işlemleri başarıyla tamamlandı!")
print(f"Eğitim seti boyutu: {X_train.shape}, Test seti boyutu: {X_test.shape}")
