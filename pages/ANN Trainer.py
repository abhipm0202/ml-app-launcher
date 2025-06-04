import ann_gui
import streamlit as st

st.sidebar.title("📥 Sample Data")
with open("sample_data/ann_sample.xlsx", "rb") as f:
    st.sidebar.download_button("Download ANN Sample", f, file_name="ann_sample.xlsx")
