import streamlit as st
from PIL import Image
import base64
from io import BytesIO

def image_to_base64(image_path, size=(300, 300)):
    if image_path is None:
        return ""
    try:
        img = Image.open(image_path).resize(size)
        buffered = BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    except Exception as e:
        return ""

def display_employee_info(name, status_class, status_text):
    st.markdown(f"""
    <div style="text-align:center;">
        <h4>{name}</h4>
        <div class="status-badge {status_class}">{status_text}</div>
    </div>
    """, unsafe_allow_html=True)

def display_result_card(name, similarity, avatar, border_color):
    img_b64 = image_to_base64(avatar)
    st.markdown(f"""
        <div style="border: 3px solid {border_color}; border-radius: 8px; padding: 5px; margin-bottom:10px; text-align: center;">
            <img src="data:image/png;base64,{img_b64}" style="width:100%; border-radius:6px;" />
            <div style="margin-top:5px;">
                <strong>{name}</strong><br>
            </div>
        </div>
    """, unsafe_allow_html=True) # <small>Similarity: {similarity:.4f}</small>