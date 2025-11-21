import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Page config (emoji + title)
st.set_page_config(page_title="Malaria Detector", page_icon="🦟", layout="centered")

# Load your trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("malaria_model_final.h5")
    return model

model = load_model()

# Beautiful header
st.title("🦟 Malaria Parasite Detector")
st.markdown("### Upload a blood smear image → get result in <1 second")
st.caption("Trained on NIH dataset • EfficientNetB3 • 96%+ accuracy")

# File uploader
uploaded_file = st.file_uploader("Drop an image here", type=["png", "jpg", "jpeg"])

if uploaded_file is not None:
    # Show uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Blood Smear", use_column_width=True)

    # Preprocess
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Analyzing..."):
        prediction = model.predict(img_array)[0][0]

    confidence = prediction if prediction > 0.5 else 1 - prediction
    confidence_pct = confidence * 100

    if prediction > 0.5:
        st.error(f"**Parasitized (Infected)**")
        st.progress(confidencence)
        st.warning(f"⚠️ {confidence_pct:.1f}% confidence → Seek medical help!")
    else:
        st.success(f"**Uninfected (Healthy)**")
        st.progress(confidence)
        st.balloons()
        st.success(f"✅ {confidence_pct:.1f}% confidence")
