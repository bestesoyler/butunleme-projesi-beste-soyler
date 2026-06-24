import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import matplotlib.pyplot as plt

# 1. Profesyonel Veri Yükleme ve Ön İşleme
print("Veri seti yükleniyor ve temizleniyor...")
# Ayrıştırıcı hatası almamak için standart okuma yapıyoruz
df = pd.read_csv("dataset.csv") 
df = df.dropna()

# Veri setindeki son sütunu otomatik hedef olarak belirliyoruz (isim farklılıklarını aşmak için)
hedef_sutun = df.columns[-1]
print(f"Hedef sütun otomatik olarak '{hedef_sutun}' olarak algılandı.")

# Öğrenimi bulandırmamak için sadece ayrılanları ve mezun olanları alıyoruz (İkili Sınıflandırma)
df = df[df[hedef_sutun] != 'Enrolled']

# Hedef değişkeni ayırma ve sayısal değere dönüştürme (Dropout = 1 (Riskli), Graduate = 0 (Risksiz))
X = df.drop(hedef_sutun, axis=1)
y = df[hedef_sutun].apply(lambda x: 1 if x == 'Dropout' else 0)

# 2. Eğitim ve Test Setlerine Bölme
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Veri Ölçeklendirme
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X.columns)
X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X.columns)

# 3. Model Eğitimi, Karşılaştırma ve Overfitting Analizi
print("\nModeller eğitiliyor ve performans metrikleri hesaplanıyor...")

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, random_state=42)
}

best_model = None
best_f1 = 0
best_model_name = ""

for name, model in models.items():
    model.fit(X_train_scaled_df, y_train)
    
    y_train_pred = model.predict(X_train_scaled_df)
    y_test_pred = model.predict(X_test_scaled_df)
    
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_f1 = f1_score(y_test, y_test_pred)
    
    print(f"\n--- {name} ---")
    print(f"Eğitim Doğruluğu: {train_acc:.4f}")
    print(f"Test Doğruluğu:   {test_acc:.4f}")
    print(f"F1 Skoru (Test):  {test_f1:.4f}")
    
    if train_acc - test_acc > 0.08:
        print(">> Uyarı: Modelde Overfitting (Aşırı Öğrenme) belirtisi var! Eğitim verisini ezberlemiş.")
    elif train_acc < 0.70:
         print(">> Uyarı: Modelde Underfitting (Eksik Öğrenme) belirtisi var! Kalıpları yakalayamamış.")
    else:
         print(">> Durum: Model dengeli öğrenmiş (Optimal Fit).")
         
    if test_f1 > best_f1:
        best_f1 = test_f1
        best_model = model
        best_model_name = name

print(f"\n🏆 En iyi model {best_model_name} seçildi ve kaydediliyor...")
joblib.dump(best_model, 'en_iyi_model.pkl')
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(X.columns, 'model_sutunlari.pkl')

# 4. SHAP ile Açıklanabilir Yapay Zeka (XAI) Entegrasyonu
print("\nSHAP değerleri hesaplanıyor ve analiz grafiği çıkarılıyor...")
X_test_sample = X_test_scaled_df.sample(min(100, len(X_test_scaled_df)), random_state=42)

if best_model_name == "Logistic Regression":
    explainer = shap.Explainer(best_model, X_train_scaled_df)
else:
    explainer = shap.TreeExplainer(best_model)

shap_values = explainer.shap_values(X_test_sample)

plt.figure(figsize=(10, 6))
if isinstance(shap_values, list):
    shap.summary_plot(shap_values[1], X_test_sample, show=False)
else:
    shap.summary_plot(shap_values, X_test_sample, show=False)

plt.title(f"{best_model_name} - SHAP Karar Açıklanabilirlik Analizi")
plt.tight_layout()
plt.savefig('shap_analiz_raporu.png', dpi=300)
print("SHAP özet grafiği 'shap_analiz_raporu.png' olarak başarıyla kaydedildi!")
print("Tüm işlemler kusursuz şekilde tamamlandı.")