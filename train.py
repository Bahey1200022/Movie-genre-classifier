import os
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn
from tqdm import tqdm
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, TaskType
from Filmdataloader import FilmdataLoader
from classifier import Classifier

# -----------------------------
# Settings
# -----------------------------
BATCH_SIZE = 16
LR = 1e-4
NUM_EPOCHS = 5
NUM_CLASSES = 19
CHECKPOINT_PATH = "clip_classifier_checkpoint.pth"
CSV_PATH = "films.csv"
POSTER_DIR = "posters"

# -----------------------------
# Load CLIP + LoRA
# -----------------------------
model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)
processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION
)
clip_model = get_peft_model(clip_model, lora_config)

# -----------------------------
# Dataset
# -----------------------------
dataset = FilmdataLoader(CSV_PATH, POSTER_DIR)

# -----------------------------
# Train/Validation/Test split
# -----------------------------
train_size = int(0.7 * len(dataset))
val_size = int(0.15 * len(dataset))
test_size = len(dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

# -----------------------------
# Custom collate for PIL images
# -----------------------------
def collate_fn(batch):
    images, labels = zip(*batch)
    return list(images), torch.stack(labels)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=collate_fn)

# -----------------------------
# Classifier & optimizer
# -----------------------------
classifier = Classifier(512, NUM_CLASSES)

optimizer = torch.optim.AdamW(
    list(clip_model.parameters()) + list(classifier.parameters()),
    lr=LR
)

loss_fn = nn.BCEWithLogitsLoss()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
clip_model.to(device)
classifier.to(device)

# -----------------------------
# Training Loop
# -----------------------------
for epoch in range(NUM_EPOCHS):
    clip_model.train()
    classifier.train()
    total_loss = 0.0
    
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{NUM_EPOCHS}"):
        optimizer.zero_grad()
        
        # Move labels to device
        labels = labels.to(device)
        
        # CLIP preprocessing
        inputs = processor(images=images, return_tensors="pt").to(device)
        
        # Get CLIP embeddings
        features = clip_model.get_image_features(**inputs)
        
        # Classifier
        logits = classifier(features)
        
        # Loss
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    
    # -----------------------------
    # Validation
    # -----------------------------
    clip_model.eval()
    classifier.eval()
    val_loss = 0.0
    with torch.no_grad():
        for images, labels in val_loader:
            labels = labels.to(device)
            inputs = processor(images=images, return_tensors="pt").to(device)
            features = clip_model.get_image_features(**inputs)
            logits = classifier(features)
            loss = loss_fn(logits, labels)
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    
    print(f"Epoch {epoch+1}/{NUM_EPOCHS} | Train Loss: {avg_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

# -----------------------------
# Save checkpoint
# -----------------------------
checkpoint = {
    "clip_model_state_dict": clip_model.state_dict(),
    "classifier_state_dict": classifier.state_dict(),
    "optimizer_state_dict": optimizer.state_dict()
}
torch.save(checkpoint, CHECKPOINT_PATH)
print(f"Checkpoint saved to {CHECKPOINT_PATH}")

# -----------------------------
# Testing (optional)
# -----------------------------
clip_model.eval()
classifier.eval()
test_loss = 0.0
with torch.no_grad():
    for images, labels in test_loader:
        labels = labels.to(device)
        inputs = processor(images=images, return_tensors="pt").to(device)
        features = clip_model.get_image_features(**inputs)
        logits = classifier(features)
        loss = loss_fn(logits, labels)
        test_loss += loss.item()
avg_test_loss = test_loss / len(test_loader)
print(f"Test Loss: {avg_test_loss:.4f}")
