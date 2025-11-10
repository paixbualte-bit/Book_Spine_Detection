import streamlit as st
from PIL import Image
from ultralytics import YOLO

# Use st.cache_resource to load the model only once
@st.cache_resource
def load_model():
    """Loads the YOLOv8 model from the 'best.pt' file."""
    # The 'best.pt' file must be in the same folder as app.py
    model = YOLO('best.pt')
    return model

# --- Page Configuration ---
st.set_page_config(
    page_title="Library Book Spine Detector",
    page_icon="📚",
    layout="wide"
)

# --- Session State Initialization ---
# We use session_state to store the history of uploaded/processed images.
if 'history' not in st.session_state:
    st.session_state.history = []

# --- Load Model ---
# Load the model (this is cached so it's fast after the first run)
try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

# --- Sidebar ---
with st.sidebar:
    st.title("📚 Book Spine Detector")
    st.markdown("---")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Upload an image of a bookshelf",
        type=["jpg", "png", "jpeg"]
    )
    
    # Result button
    result_button = st.button("Detect Spines")
    st.markdown("---")
    st.markdown("Built with YOLOv8 & Streamlit")

# --- Main Page Content ---
st.title("Library Book Spine Detection")
st.markdown("Upload an image, then click the **'Detect Spines'** button in the sidebar.")

# Main logic
if result_button and uploaded_file is not None:
    # Load the uploaded image
    image = Image.open(uploaded_file)

    # Perform detection
    with st.spinner('Detecting book spines...'):
        results = model(image)

    # Plot the results
    # results[0].plot() returns a NumPy array (BGR)
    res_plotted = results[0].plot()
    
    # Convert BGR (from OpenCV) to RGB (for PIL/Streamlit)
    res_plotted_rgb = Image.fromarray(res_plotted[..., ::-1])

    # Add to history (at the beginning of the list)
    st.session_state.history.insert(0, (image, res_plotted_rgb))

    # Display the result on the main page
    st.header("Detection Result")
    st.image(res_plotted_rgb, caption="Detected Book Spines", use_column_width=True)

elif uploaded_file is not None and not result_button:
    # Show the uploaded image before processing
    st.header("Your Uploaded Image")
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

# --- History Section ---
if st.session_state.history:
    st.markdown("---")
    st.header("Detection History")
    
    # Display history in two columns (original vs. detected)
    for i, (original, detected) in enumerate(st.session_state.history):
        st.markdown(f"**History Item {i+1}**")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Original", use_column_width=True)
        with col2:
            st.image(detected, caption="Detected", use_column_width=True)
        st.markdown("---")