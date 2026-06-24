## 🎥 Proje Tanıtım Videosu
[Projenin Çalışma Mantığını İzlemek İçin Tıklayın](https://youtu.be/d_PzQYTcFIA)

# AURA: MCBÜ Akademik Karar Destek Sistemi
**Geliştirici:** Beste Söyler | **Bölüm:** Veri Bilimi ve Analitiği

## Proje Hakkında
MCBÜ öğrencileri için geliştirilen, makine öğrenmesi (Random Forest/Logistic Regression) destekli akademik risk öngörü ve fırsat radarıdır. Sistem, öğrencilerin "okulu bırakma" (dropout) riskini analiz ederken, aynı zamanda GANO simülasyonu ile öğrencilere kariyer ve akademik fırsatlar (Erasmus, TÜBİTAK, Hackathon) önerir.

## Özgün Değerler
- **Senaryo Modu:** Öğrenciler ders notlarını simüle ederek risk değişimlerini anlık görebilir.
- **Akademik Fırsat Radarı:** Öğrencinin not ortalamasına göre MCBÜ bünyesindeki güncel Erasmus ve sanayi iş birliği fırsatlarını filtreler.
- **Şeffaf Kararlar:** SHAP kütüphanesi ile modelin verdiği kararlar "kara kutu" olmaktan çıkarılmıştır.

## Teknolojiler
- Python 3.x, Streamlit, Pandas, Scikit-learn, SHAP, Matplotlib

## Nasıl Çalıştırılır?
1. Gerekli kütüphaneleri yükleyin: `pip install -r requirements.txt`
2. Uygulamayı başlatın: `streamlit run arayuz.py`

## Proje İçerik Tablosu

| Sayfa / Dosya | İçerik |
| :--- | :--- |
| **arayuz.py** | Senaryo modu, GANO simülasyonu ve Fırsat Radarı arayüzü |
| **model_egitimi.py** | Random Forest/Logistic Regression model eğitim süreçleri |
| **veri_hazirlama.py** | Kaggle veri seti temizleme ve öznitelik mühendisliği |
| **Butunleme_Raporu.pdf** | Projenin akademik ve teknik kapsam raporu |

## 📅 Proje Geliştirme Günlüğü (3 Günlük Süreç)
- **1. Gün (23.06.2026):** Veri seti analizi, temizleme ve temel makine öğrenmesi model eğitimi çalışmaları.
- **2. Gün (24.06.2026):** Arayüz (Streamlit) geliştirme, simülasyon mantığı ve model entegrasyonu.
- **3. Gün (25.06.2026):** Raporlama, XAI (SHAP) analizi dokümantasyonu ve tanıtım videosu hazırlığı.
