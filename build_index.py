import os
import numpy as np
import faiss
import torch
import torchvision.transforms as T
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from core_logic import face_recognition_model as model, transform, crop_center_square

# 1. Setup Configuration
DATASET_DIR = "Dataset"
INDEX_FILE = "facenet_features.index"
LABEL_FILE = "facenet_label_map.npy"

# Check if dataset exists
if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)
    print(f"Created {DATASET_DIR} folder. Please put employee images here.")
    exit()

# 2. Prepare Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = model.to(device) # Move imported model to device

# 3. Process Images
embeddings = []
names = []

print("Starting to process images...")

valid_extensions = {".jpg", ".jpeg", ".png"}

for filename in os.listdir(DATASET_DIR):
    ext = os.path.splitext(filename)[1].lower()
    if ext not in valid_extensions:
        continue
    
    name = os.path.splitext(filename)[0]
    if name.startswith("Avatar_"):
        name = name.replace("Avatar_", "")
    
    image_path = os.path.join(DATASET_DIR, filename)
    
    try:
        # Load and preprocess
        img = Image.open(image_path).convert('RGB')
        img = crop_center_square(img)
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        # Get embedding
        with torch.no_grad():
            embedding = model(img_tensor).cpu().numpy()
            
        embeddings.append(embedding.flatten())
        names.append(name)
        print(f"Processed: {name}")
        
    except Exception as e:
        print(f"Error processing {filename}: {e}")

# 4. Create and Save Index
if len(embeddings) == 0:
    print("No images found or processed! Please add images to Dataset/ folder.")
else:
    embeddings_np = np.array(embeddings).astype('float32')
    
    # Normalize vectors for Cosine Similarity
    faiss.normalize_L2(embeddings_np)
    
    # Create FAISS index (Inner Product used for Cosine Similarity)
    dimension = embeddings_np.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings_np)
    
    # Save
    faiss.write_index(index, INDEX_FILE)
    np.save(LABEL_FILE, np.array(names))
    
    print("\nSUCCESS!")
    print(f"Saved {len(names)} faces to {INDEX_FILE}")
    print(f"Saved label map to {LABEL_FILE}")