import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import streamlit as st
from PIL import Image
import cv2
import time
import numpy as np
from ui_components import display_employee_info, display_result_card
from core_logic import image_to_feature, search_similar_faces, get_employee_avatar, crop_center_square
import config

# Initialize Session State
if 'checkin_status' not in st.session_state:
    st.session_state.checkin_status = {}
    # Load existing employees if available
    if os.path.exists('facenet_label_map.npy'):
        try:
            employees = np.load('facenet_label_map.npy')
            for emp in employees:
                st.session_state.checkin_status[emp] = False
        except Exception as e:
            st.error(f"Failed to load employee list: {e}")

if 'all_matches' not in st.session_state:
    st.session_state.all_matches = []

if 'query_embedding' not in st.session_state:
    st.session_state.query_embedding = None

if 'captured_image' not in st.session_state:
    st.session_state.captured_image = None

if 'captured_clicked' not in st.session_state:
    st.session_state.captured_clicked = False
    
if 'matching_result' not in st.session_state:
    st.session_state.matching_result = None

if 'matching_distance' not in st.session_state:
    st.session_state.matching_distance = None

if 'matching_avatar' not in st.session_state:
    st.session_state.matching_avatar = None

# Main Layout
st.markdown("<h1 style='text-align: center;'>🧑‍💼 Face-Based Employee Check-in System</h1>", unsafe_allow_html=True)
col1, col2 = st.columns([3, 2], gap="large")

with col1:
    # Employee List Section
    st.markdown("### 👥 Employee List")
    
    if not st.session_state.checkin_status:
        st.warning("No employee data available. Please ensure the database is properly set up.")
    else:
        # Create a grid of employee cards
        cols = st.columns(3)
        for i, (name, checked) in enumerate(st.session_state.checkin_status.items()):
            with cols[i % 3]:
                avatar = get_employee_avatar(name)
                print(avatar, i)
                status_class = "checked-in" if checked else "not-checked"
                status_text = "CHECKED IN" if checked else "NOT CHECKED"
                if avatar and os.path.exists(avatar):
                    image = Image.open(avatar)
                else:
                    # Create a dummy image or handle missing avatar
                    image = Image.new('RGB', (300, 300), color = (200, 200, 200))

                image = image.resize((300, 300))
                
                st.image(image, use_container_width=True)

                display_employee_info(name, status_class, status_text)

with col2:
    # Check-in Section
    st.markdown("### 📸 Employee Check-in")
    
    # Camera capture button
    if not st.session_state.get('capture_clicked', False):
        if st.button("Open Camera for Check-in", use_container_width=True, type="primary"):
            st.session_state.capture_clicked = True
            st.rerun()
    else:
        if st.button("❌ Cancel Check-in", use_container_width=True):
            st.session_state.capture_clicked = False
            st.rerun()
    
    if st.session_state.get('capture_clicked', False):
        st.info("Checking in...")
        st.session_state.captured_image = None
        camera_placeholder = st.empty()
        note_placeholder = st.empty()

        cap = cv2.VideoCapture(0)
        captured_frame = None

        if not cap.isOpened():
            st.warning("📷 Could not access camera. Please upload an image instead.")
            uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                captured_image = Image.open(uploaded_file).convert("RGB")
                captured_image = crop_center_square(captured_image)
                st.session_state.captured_image = captured_image
        else:
            face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
            note_placeholder.markdown(
                    "<h3 style='text-align:center;color:#4CAF50;'>😊 Smile to capture</h3>", 
                    unsafe_allow_html=True
                )
            
            captured_frame = None
            while True:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to capture frame")
                    break

                # Process image
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
                is_smiling = False
                for (x, y, w, h) in faces:
                    roi_gray = gray[y:y+h, x:x+w]
                    smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=10)
                    if len(smiles) > 0:
                        is_smiling = True
                        cv2.putText(frame_rgb, "Smile!", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                        cv2.rectangle(frame_rgb, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        break # Only need one smiling face

                # Update image display
                camera_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                if is_smiling:
                    captured_frame = frame_rgb
                    break

                time.sleep(0.05)

            cap.release()
            camera_placeholder.empty()
            note_placeholder.empty()

            if captured_frame is not None:
                captured_image = Image.fromarray(captured_frame)
                captured_image = crop_center_square(captured_image)
                st.session_state.captured_image = captured_image

        # If we have a captured image (from camera or upload)
        if st.session_state.get('captured_image'):
            with st.spinner("🔍 Finding match..."):
                query_embedding = image_to_feature(st.session_state.captured_image)
                st.session_state.query_embedding = query_embedding

                matches = search_similar_faces(query_embedding, k=5, threshold=0.3)
                st.session_state.all_matches = matches

                if matches:
                    best_match_name, best_distance = matches[0]
                    st.session_state.checkin_status[best_match_name] = True
                    st.session_state.matching_result = best_match_name
                    st.session_state.matching_distance = best_distance
                    st.session_state.matching_avatar = get_employee_avatar(best_match_name)
                    st.session_state.capture_clicked = False
                    st.rerun()
                else:
                    st.session_state.matching_result = None
                    st.session_state.capture_clicked = False
                    st.rerun()

    # Show captured image if available
    if st.session_state.captured_image is not None:
        st.markdown("---")
        st.markdown("### 📷 Captured Image")
        st.image(st.session_state.captured_image, use_container_width=True)
        
        # Show matching results
        if st.session_state.matching_result:
            st.success(f"""
            ✅ **{st.session_state.matching_result}** checked in!
            """) # Similarity: {st.session_state.matching_distance:.4f}
            
            # Show top matches
            st.markdown("### 🏆 Top Matches")
            inner_cols = st.columns(2)  # Create two columns inside col2

            for i, (name, similarity) in enumerate(st.session_state.all_matches[:5]):
                avatar = get_employee_avatar(name)
                border_color = "#4CAF50" if i == 0 else "#FFC107"

                with inner_cols[i % 2]:  # Alternate between the two inner columns
                    display_result_card(name, similarity, avatar, border_color)

        elif st.session_state.matching_result is None:
            st.error("No matches found.") # No matches found above the confidence threshold