import torch
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
from classifier import Classifier
import pandas as pd

# -----------------------------
# Load genre mapping
# -----------------------------
ids_df = pd.read_csv("ids.csv")
id_to_genre = {row["id"]: row["genre"] for _, row in ids_df.iterrows()}
num_classes = len(id_to_genre)

# -----------------------------
# Load models
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CLIP + LoRA
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
clip_model = get_peft_model(clip_model, lora_config)
clip_model.to(device)
clip_model.eval()

# Classifier
classifier = Classifier(512, num_classes)
classifier.to(device)
classifier.eval()

# Load checkpoint
checkpoint = torch.load("clip_classifier_checkpoint.pth", map_location=device)
clip_model.load_state_dict(checkpoint["clip_model_state_dict"])
classifier.load_state_dict(checkpoint["classifier_state_dict"])

# -----------------------------
# Processor
# -----------------------------
processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

# -----------------------------
# Load your test image
# -----------------------------
image_path = "posters/928358.jpg"
image = Image.open(image_path).convert("RGB")

# -----------------------------
# Preprocess & predict
# -----------------------------
inputs = processor(images=[image], return_tensors="pt").to(device)
with torch.no_grad():
    features = clip_model.get_image_features(**inputs)
    logits = classifier(features)                # raw logits
    probs = torch.sigmoid(logits)               # convert to probabilities

# -----------------------------
# Threshold to get predicted genres
# -----------------------------
threshold = 0.5
predicted_indices = (probs > threshold).nonzero(as_tuple=True)[1].tolist()
predicted_genres = [id_to_genre[i] for i in predicted_indices]

print("Predicted Genres:", predicted_genres)
