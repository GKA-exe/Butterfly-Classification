import streamlit as st
from PIL import Image
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import pandas as pd
import os

# Configure the page for a neat UI
st.set_page_config(page_title="Butterfly & Moth Classifier", layout="wide")
st.markdown(
    """
    <style>
    .main {
        background-color: #F0F2F6;
        padding: 2rem;
    }
    h1 {
        color: #2C3E50;
    }
    </style>
    """, unsafe_allow_html=True
)

st.title("Butterfly & Moth Classification Model")
st.write("Upload an image and the model will classify it among 100 classes.")

# Define the model architecture to match your training code
def load_model():
    # Load the pretrained VGG19 model
    model = models.vgg19(weights=models.VGG19_Weights.DEFAULT)
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Replace the classifier with your custom classifier
    classifier = torch.nn.Sequential(
        torch.nn.Linear(25088, 2048),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(2048, 2048),
        torch.nn.ReLU(),
        torch.nn.Dropout(p=0.3),
        torch.nn.Linear(2048, 100),
        torch.nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    
    # Load the checkpoint (ensure the path is correct)
    checkpoint_path = "./checkpoint.pt"
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint["state_dict"])
        st.write("Checkpoint Loaded Successfully.")
    else:
        st.error("Checkpoint file not found!")
    
    model.eval()
    return model

# Use st.cache_resource for caching model resources.
@st.cache_resource
def get_model():
    return load_model()

model = get_model()

# Use st.cache_data for caching data objects (species mapping)
@st.cache_data
def load_species_mapping():
    csv_path = "./data/butterflies and moths.csv"
    df = pd.read_csv(csv_path)
    # Create mapping: assuming columns "class id" and "labels"
    mapping = df[['class id', 'labels']].drop_duplicates().set_index('class id')['labels'].to_dict()
    return mapping

species_mapping = load_species_mapping()

# Define image transformation based on your test_transforms from training
transform = transforms.Compose([
    transforms.Resize(225),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Open, convert, and display the image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Preprocess the image
    image_tensor = transform(image).unsqueeze(0)  # add batch dimension
    
    if st.button("Classify"):
        with torch.no_grad():
            output = model(image_tensor)
            # Since your model outputs log_softmax, exponentiate to get probabilities
            probabilities = torch.exp(output)
            # Retrieve the top 5 predictions
            topk_prob, topk_indices = torch.topk(probabilities, k=5, dim=1)
            
            st.write("### Top Predictions:")
            for prob, idx in zip(topk_prob[0], topk_indices[0]):
                # Use the mapping from the CSV; if not found, show the class index
                label = species_mapping.get(idx.item(), f"Class {idx.item()}")
                st.write(f"**{label}** with confidence: {prob.item():.2f}")
            
            # st.balloons()
