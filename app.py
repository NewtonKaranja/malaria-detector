import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import urllib.request

st.set_page_config(page_title="Malaria Detector", page_icon="🦟", layout="centered")

# Auto-download model once (85 MB → only first visitor waits 20 sec)
MODEL_PATH = "malaria_model.h5"
if not os.path.exists(MODEL_PATH):
    with st.spinner("First launch – downloading model (20 sec)..."):
        url = "https://huggingface.co/greyarea/malaria-detector/resolve/main/malaria_model_final.h5"
        urllib.request.urlretrieve(url, MODEL_PATH)

@st.cache_resource
def load_model():
    return tf.keras.models.load_model(MODEL_PATH)

model = load_model()

st.title("Malaria Parasite Detector")
st.markdown("**Upload a blood smear → instant diagnosis**")
st.caption("EfficientNetB3 • Real NIH dataset • 96%+ accuracy")

file = st.file_uploader("Drop image here", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)
    
    x = img.resize((224, 224))
    x = np.array(x) / 255.0
    x = x[None, ...]
    
    pred = model.predict(x, verbose=0)[0][0]
    conf = pred if pred > 0.5 else 1-pred
    pct = conf * 100
    
    if pred > 0.5:
        st.error(f"PARASITIZED – {pct:.1f}% confidence")
        st.warning("Seek medical attention immediately!")
    else:
        st.success(f"UNINFECTED – {pct:.1f}% confidence")
        st.balloons()
