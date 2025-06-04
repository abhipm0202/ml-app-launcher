import lstm_gui
import streamlit as st

st.sidebar.title("📥 Sample Data")
with open("sample_data/lstm_sample.xlsx", "rb") as f:
    st.sidebar.download_button("Download LSTM Sample", f, file_name="lstm_sample.xlsx")
