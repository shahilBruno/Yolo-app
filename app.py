import streamlit as st
import torch
from PIL import Image
import numpy as np
import os 

# 1. Page Setup
st.set_page_config(page_title="YOLO Native Loader", layout="wide")
st.title("🎯 YOLO via PyTorch")

# 2. The "Native" Load
# This pulls the model logic directly from GitHub. 
@st.cache_resource
def load_yolo():
    local_weights = 'yolov5s.pt'
    repo = 'ultralytics/yolov5'
    if os.path.exists(local_weights):
        st.sidebar.success("✅ Loaded weights")
        model = torch.hub.load(repo, 'custom', path=local_weights)
        return model 
    st.sidebar.warning("🌐 Downloading weights...")
    model = torch.hub.load(repo, 'yolov5s', pretrained=True)
    return model
# otherwise download always
# @st.cache_resource
# def load_yolo():
#     # 'ultralytics/yolov5' is the repo, 'yolov5s' is the model size
#     model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
#     return model
model = load_yolo()

# 3. Settings sidebar
st.sidebar.header("Inference Settings")
conf_val = st.sidebar.slider("Confidence", 0.0, 1.0, 0.4)
model.conf = conf_val  # You can set settings directly on the hub model

# 4. File Uploader
img_file = st.file_uploader("Upload Image here", type=['jpg', 'png', 'jpeg', 'webp'])

if img_file:
    img = Image.open(img_file)
    
    # 5. Run Inference
    # The Hub model is very flexible; it accepts PIL images directly
    results = model(img)
    
    # # use two columns for original and detection
    # col1, col2 = st.columns(2)
    # with col1:
    #     st.subheader("Original")
    #     st.image(img, use_container_width=True)
    # with col2:
    #     st.subheader("Detections")
    #     # .render() returns a list of images (numpy arrays) with boxes drawn
    #     rendered_img = results.render()[0]
    #     st.image(rendered_img, use_container_width=True)

    # only the detection
    st.subheader("Detections")
    # .render() returns a list of images (numpy arrays) with boxes drawn
    rendered_img = results.render()[0]
    st.image(rendered_img, use_container_width=True)

    # 6. Technical Breakdown (The Data)
    st.divider()
    st.subheader("Raw Detection Data")
    
    # .pandas() is a helper method provided by the Hub model
    # It gives you a clean table of [xmin, ymin, xmax, ymax, confidence, class, name]
    df = results.pandas().xyxy[0] 
    
    if not df.empty:
        st.dataframe(df)
    else:
        st.warning("No objects found above threshold.")