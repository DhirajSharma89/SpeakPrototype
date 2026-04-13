import streamlit as st
import torch

st.title("Speak Prototype")

st.write("Torch version:", torch.__version__)
st.write("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    st.write("GPU:", torch.cuda.get_device_name(0))