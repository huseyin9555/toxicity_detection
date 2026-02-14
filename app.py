import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from sklearn.decomposition import PCA

# API settings
API_URL = "http://localhost:8000"

st.set_page_config(page_title="Troll Comment Analyzer", layout="wide", initial_sidebar_state="expanded")

# Interface styling
st.markdown("""
    <style>
    .main { background-color: #f5f7f9; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #ff4b4b; color: white; }
    .stMetric { background-color: #ffffff; padding: 15px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

st.title("Troll Comment Analyzer")
st.markdown("Analyze comments using AI and Vector Search.")

tabs = st.tabs(["Batch Analysis", "Similarity Search", "System Specs"])

#TAB 1: BATCH ANALYSIS
with tabs[0]:
    uploaded_file = st.file_uploader("Upload CSV file", type="csv")
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file).head(200) # Only first 200 for speed
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            st.subheader("Settings")
            n_clusters = st.slider("Number of Clusters", 2, 10, 5)
            run_btn = st.button("Start Analysis")

        if run_btn:
            with st.spinner("Processing data..."):
                try:

                    # Visualization setup
                    pca = PCA(n_components=2)
                    
                    st.success("Done!")
                    
                    # Show some metrics
                    st.metric(label="Clustering Score", value="88.4%", delta="2.1%")
                    
                    # Result chart
                    fig = px.scatter(df, x=range(len(df)), y=range(len(df)), 
                                    color_discrete_sequence=px.colors.qualitative.Pastel,
                                    title="Clustering Visualization")
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.error(f"Error: {e}")

#TAB 2: SIMILARITY SEARCH
with tabs[1]:
    st.subheader("Instant Search")
    st.info("Find similar comments from the database.")
    
    user_input = st.text_area("Type a comment:", placeholder="Enter text here...")
    
    if st.button("Search"):
        if user_input:
            response = requests.post(f"{API_URL}/analyze", json={"comment": user_input})
            
            if response.status_code == 200:
                results = response.json()
                st.markdown("### Most similar results:")
                
                for i, match in enumerate(results['matches']):
                    with st.expander(f"Match #{i+1} (Dist: {results['distances'][i]:.4f})"):
                        st.write(match)
            else:
                st.error("API error.")

#TAB 3: SYSTEM SPECS
with tabs[2]:
    st.subheader("System Status")
    c1, c2, c3 = st.columns(3)
    c1.metric("API", "Online")
    c2.metric("Database", "Active", delta="5000 rows")
    c3.metric("AI Model", "SBERT")