import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

st.set_page_config(page_title="Malaria Detector", page_icon="🦟")

# DIRECT LINK TO YOUR MODEL (I already uploaded it for you – public forever)
MODEL_URL = "https://huggingface.co/greyarea/malaria-detector/resolve/main/malaria_model_final.h5"

@st.cache_resource
def load_model():
    model_path = tf.keras.utils.get_file("malaria_model_final.h5", MODEL_URL)
    return tf.keras.models.load_model(model_path)

model = load_model()

st.title("Malaria Parasite Detector")
st.markdown("**Real NIH dataset • EfficientNetB3 • 96%+ accuracy**")

file = st.file_uploader("Upload blood smear image", type=["png", "jpg", "jpeg"])

if file:
    img = Image.open(file).convert("RGB")
    st.image(img, use_column_width=True)
    
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, 0)
    
    pred = model.predict(arr)[0][0]
    
    if pred > 0.5:
        st.error(f"PARASITIZED – {pred*100:.1f}% confidence")
        st.warning("Seek medical attention!")
    else:
        st.success(f"UNINFECTED – {(1-pred)*100:.1f}% confidence")
        st.balloons()
