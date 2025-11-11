import streamlit as st
from transformers import pipeline
from PIL import Image

st.set_page_config(page_title="AI that can see - Image Classification")
st.title("AI that can see - Image Classification")


@st.cache_resource
def load_model():
    # Download and cache the model on first run
    return pipeline("image-classification", model="microsoft/resnet-50")

classifier = load_model()

def predict(pil_image: Image.Image):
    results = classifier(pil_image)
    return [{"label": r["label"], "score": float(r["score"])} for r in results]


st.sidebar.title("AI that can see - Image Classification")
st.sidebar.header("About us")
st.sidebar.info(
    """
    This app is developed by Hemant Tyagi.
    This is a simple image classification app built using Streamlit.
    Model used: microsoft/resnet-50 from Hugging Face.
    """
)

st.header("Upload an image or use your camera to classify it")

uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
camera_file = st.camera_input("Take a picture")

input_image = None
if uploaded_file is not None:
    try:
        input_image = Image.open(uploaded_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open uploaded image: {e}")
elif camera_file is not None:
    try:
        input_image = Image.open(camera_file).convert("RGB")
    except Exception as e:
        st.error(f"Could not open camera image: {e}")

if input_image is not None:
    st.image(input_image, caption="Input image", use_column_width=True)
    with st.spinner("Classifying..."):
        results = predict(input_image)

    st.markdown("**Top predictions**")
    for r in results:
        st.write(f"{r['label']}: {r['score']:.4f}")

    try:
        st.bar_chart({r['label']: r['score'] for r in results})
    except Exception:
        pass
else:
    st.info("Please upload an image or take a picture with your camera.")


