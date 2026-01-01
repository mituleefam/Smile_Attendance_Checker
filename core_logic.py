import os
import numpy as np
import faiss
from PIL import Image
from tqdm import tqdm
import torch
from torchvision import transforms
from facenet_pytorch import InceptionResnetV1

# Setup Model & Transform
face_recognition_model = InceptionResnetV1(pretrained='vggface2').eval()

# Resize image
transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

def crop_center_square(image):
    width, height = image.size
    size = min(width, height)
    left = (width - size) / 2
    top = (height - size) / 2
    right = (width + size) / 2
    bottom = (height + size) / 2
    return image.crop((left, top, right, bottom))

def image_to_feature(image_input, model=None):
    """Convert image to face embedding using a pre-trained model"""
    if model is None:
        model = face_recognition_model
    
    if isinstance(image_input, str):
        img = Image.open(image_input).convert('RGB')
    elif isinstance(image_input, Image.Image):
        img = image_input.convert('RGB')
    else:
        raise ValueError("Input must be a file path or PIL Image")
        
    img_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        features = model(img_tensor)
    return features.squeeze().numpy()

def search_similar_faces(query_input, k=5, threshold=0.4):
    index = faiss.read_index('facenet_features.index')
    label_map = np.load('facenet_label_map.npy')
    
    if isinstance(query_input, np.ndarray):
        query_vector = query_input
    else:
        query_vector = image_to_feature(query_input, face_recognition_model)
    
    # Normalize query vector for Cosine Similarity
    faiss.normalize_L2(query_vector.reshape(1, -1))
        
    similarities, indices = index.search(np.array([query_vector]), k)

    matches = []
    for i in range(k):
        if indices[0][i] == -1: continue # Handle empty index cases
        
        employee_name, similarity = label_map[indices[0][i]], similarities[0][i]
        if similarity < threshold:
            break
        matches.append((employee_name, similarity))
    return matches

def get_employee_avatar(employee_name):
    avatar_path_jpg = f"./Dataset/Avatar_{employee_name}.jpg"
    avatar_path_JPG = f"./Dataset/Avatar_{employee_name}.JPG"

    if os.path.exists(avatar_path_jpg):
        return avatar_path_jpg
    elif os.path.exists(avatar_path_JPG):
        return avatar_path_JPG
    else:
        return None