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
