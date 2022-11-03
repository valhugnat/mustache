import streamlit as st
import numpy as np
from picsellia import Client 
from PIL import Image, ExifTags
import cv2
from pillow_heif import register_heif_opener


def format_predictions(predictions, threshold=0.55):
    image = cv2.imread('img.jpg', cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    for i, bbox in enumerate(predictions["detection_boxes"]):
        if predictions["detection_scores"][i] > threshold/100:
            x, y, w, h = bbox
            cv2.rectangle(image, (y, x) , (h, w), (255,0,0), 2)
            cv2.putText(image, 'Mustache 🥸', (y, x-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
    return image

if __name__ == "__main__":
    register_heif_opener()

    st.title('Happy Movember to all of you 😃 🥸')
    st.text('Ok, the game is simple, upload a selfie now, and enjoy 😃')
    st.text('Do not hesitate to screenshot and share the selfie to promote Movember')
    deployment_name= "mustache-detection" #st.text_input(label="Please enter your deployment name")
    model = Client(organization_name='ValentinP').get_deployment(deployment_name)
    img_file_buffer = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "HEIC"])
    if img_file_buffer is not None:
        image = Image.open(img_file_buffer).convert('RGB')
        
        image.save('img.jpg')
        predictions = model.predict('img.jpg')
        conf = st.slider('CONFIDENCE?', 0, 100, 1)
        st.image(
            format_predictions(predictions=predictions, threshold=conf),
            use_column_width=True,
        )
        
    
