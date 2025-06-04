import cnn_gui
import streamlit as st

st.sidebar.title("📥 Sample Data")
with open("sample_data/cnn_sample.zip", "rb") as f:
    st.sidebar.download_button("Download CNN Sample", f, file_name="cnn_sample.zip")
