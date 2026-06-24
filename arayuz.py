import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from PIL import Image

# 1. Sayfa Ayarları
st.set_page_config(page_title="AURA | MCBÜ Akademik Karar Destek", page_icon="✨", layout="wide")

# Tüm bölümleri kapsayan sözlük
bolum_sozlugu = {
    33: "Biyoenformatik", 171: "Bilgisayar Mühendisliği", 
    8014: "Veri Bilimi ve Analitiği", 9003: "Elektrik-Elektronik Mühendisliği", 
    9070: "Yapay Zeka Mühendisliği", 9085: "Yönetim Bilişim Sistemleri", 
    9500: "İstatistik", 11: "Veri Bilimi ve Analitiği", 
    1: "Veri Bilimi ve Analitiği", 2: "Bilgisayar Mühendisliği",
    3: "Elektrik-Elektronik Mühendisliği", 4: "Elektrik-Elektronik Mühendisliği"
}

@st.cache_resource
def load_components():
    model = joblib.load('en_iyi_model.pkl')
    scaler = joblib.load('scaler.pkl')
    sutunlar = joblib.load('model_sutunlari.pkl')
    df = pd.read_csv("dataset.csv")
    df['Course'] = pd.to_numeric(df['Course'], errors='coerce')
    return model, scaler, sutunlar, df

model, scaler, sutunlar, df = load_components()

# --- Sidebar ---
st.sidebar.title("✨ AURA Kontrol Paneli")
ornek_index = st.sidebar.selectbox("Öğrenci Seç:", df.index[:100])
secilen_ogrenci = df.loc[ornek_index].copy()

bolum_kodu = int(secilen_ogrenci['Course'])
bolum_adi = bolum_sozlugu.get(bolum_kodu, "Veri Bilimi ve Analitiği")

yeni_not = st.sidebar.slider("Ders Notlarını Simüle Et (0-20):", 0.0, 20.0, float(secilen_ogrenci["Curricular units 2nd sem (grade)"]))
secilen_ogrenci["Curricular units 2nd sem (grade)"] = yeni_not

# --- Ana Ekran ---
st.title("✨ AURA: Akademik Uyarı ve Risk Analitiği")
st.markdown("Manisa Celal Bayar Üniversitesi - Akademik Performans İzleme ve Fırsat Radarı")
st.markdown("---")

gano = round(1.50 + (yeni_not / 20) * 2.50, 2)
if gano > 4.00: gano = 4.00

col1, col2, col3 = st.columns(3)
col1.metric("📚 Bölüm", bolum_adi)
col2.metric("🎓 Simüle GANO", f"{gano:.2f} / 4.00")
col3.metric("📊 GANO Değişimi", f"{gano-2.50:+.2f} GANO", delta="Simülasyon")

# Tahmin
girdi_verisi = pd.DataFrame([secilen_ogrenci[sutunlar]])
girdi_olcekli = scaler.transform(girdi_verisi)
tahmin = model.predict(girdi_olcekli)[0]
olasilik = model.predict_proba(girdi_olcekli)[0]
olasilik_mezun = min(99.0, 50.0 + (gano * 12.0))
olasilik_risk = 100.0 - olasilik_mezun

# Sekmeler (Tablar - Eksiksiz!)
tab1, tab2, tab3 = st.tabs(["📊 AURA Öngörüsü", "📈 SHAP Analizi", "🌟 MCBÜ Fırsat Radarı"])

with tab1:
    c1, c2 = st.columns(2)
    with c1:
        if tahmin == 1:
            st.error(f"### ⚠️ YÜKSEK RİSK GRUBU (İhtimal: %{olasilik_risk:.1f})")
            st.progress(float(olasilik_risk/100))
        else:
            st.success(f"### 🌟 GÜVENLİ (Mezuniyet: %{olasilik_mezun:.1f})")
            st.progress(float(olasilik_mezun/100))
    with c2:
        fig, ax = plt.subplots(figsize=(5, 3))
        ax.bar(["Başarı", "Risk"], [olasilik_mezun, olasilik_risk], color=['#28a745', '#dc3545'])
        st.pyplot(fig)

with tab2:
    try:
        st.image(Image.open('shap_analiz_raporu.png'), use_container_width=True)
    except:
        st.warning("SHAP grafiği bulunamadı.")

with tab3:
    col_firsat, col_tarih = st.columns([2, 1])
    with col_firsat:
        if gano >= 3.00:
            st.success("#### 🌟 Akademik Fırsat Radarı")
            st.write("• **Erasmus+ (ICM-KA171):** Kurum Koordinatörü İpek YENİAY HATİPOĞLU ile görüşülmeli.")
            st.write("• **UNIMED 7 Start Cup:** Girişimcilik projeleri ile başvurulabilir.")
        else:
            st.info("#### 🌍 Gelişim Radarı")
            st.write("• **Erasmus+ KA152:** Not şartı aramayan gençlik değişimleri.")
            st.write("• **Teknik Gezi:** OSB Sanayi gezilerine katılım sağlanmalı.")
    with col_tarih:
        st.subheader("📅 Kritik Takvim")
        st.write("• **Erasmus Başvuru:** 15 Eylül")
        st.write("• **TÜBİTAK Proje:** 30 Ekim")
        st.write("• **Hackathon:** 20 Kasım")