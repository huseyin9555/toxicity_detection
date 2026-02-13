import streamlit as st
import pandas as pd
from src.processor import clean_text
from src.model_logic import get_embeddings, run_clustering

st.set_page_config(page_title="Troll Detector", layout="wide")
st.title("🛡️ Wikipedia Troll & Yorum Kümeleme")

uploaded_file = st.file_uploader("CSV dosyanızı yükleyin (comment_text sütunu içermeli)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file).head(500) # Hız için ilk 500 satır
    st.write(f"Toplam {len(df)} yorum analiz ediliyor...")
    
    with st.spinner('Metinler temizleniyor ve vektörize ediliyor...'):
        df['cleaned_text'] = df['comment_text'].apply(clean_text)
        embeddings = get_embeddings(df['cleaned_text'].tolist())
        df['cluster'] = run_clustering(embeddings)
        
    st.success("Analiz Tamamlandı!")
    st.dataframe(df[['comment_text', 'cluster']])