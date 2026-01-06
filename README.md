# Smile Attendance Checker

An AI-powered attendance tracking system that uses facial recognition to identify employees and mark their attendance. Built with **Streamlit**, **FaceNet (PyTorch)**, and **FAISS**.

##  Features

- **Real-time Face Recognition:** Uses webcam input to detect and identify faces.
- **Efficient Matching:** specific FAISS (Facebook AI Similarity Search) index for fast face vector comparison.
- **Attendance Tracking:** Visual dashboard to show employee check-in status.
- **Easy Management:**  Simple workflow to add new employees by dropping images into a folder.

##  Tech Stack

- **Frontend:** Streamlit
- **ML/AI:** PyTorch, InceptionResnetV1 (FaceNet), FAISS
- **Image Processing:** OpenCV, PIL
- **Language:** Python 3.8+

##  Installation & Setup

### 1. Clone the repository
```bash
git clone <repository_url>
cd Smile_Attendance_Checker
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Mac/Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

##  Data Preparation

1.  Navigate to the `Dataset` folder.
2.  Add employee images.
    *   **Filename Format:** The filename will be used as the employee's name (e.g., \`Avatar_John_Doe.jpg\` -> Name: \`John Doe\`).
    *   Ensure images have clear visibility of the face.

##  Usage

### 1. Build the Facial Recognition Index
Before running the app for the first time or after adding new images, you must build the features index:

```bash
python build_index.py
```
*This script processes images in the \`Dataset/\` folder, extracting facial features and saving them to \`facenet_features.index\`.*

### 2. Run the Application
Start the Streamlit web interface:

```bash
streamlit run app.py
```