import os
from flask import Flask, render_template_string, request, send_from_directory
import torch
from transformers import CLIPProcessor, CLIPModel
from peft import LoraConfig, get_peft_model, TaskType
from PIL import Image
import pandas as pd
from classifier import Classifier

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "uploads"
os.makedirs("uploads", exist_ok=True)

# Load genre mapping
ids_df = pd.read_csv("ids.csv")
id_to_genre = {row["id"]: row["genre"] for _, row in ids_df.iterrows()}
num_classes = len(id_to_genre)

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model_name = "openai/clip-vit-base-patch32"
clip_model = CLIPModel.from_pretrained(model_name)

lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1,
    bias="none",
    task_type=TaskType.FEATURE_EXTRACTION,
)

clip_model = get_peft_model(clip_model, lora_config)
clip_model.to(device).eval()

classifier = Classifier(512, num_classes).to(device)
classifier.eval()

checkpoint = torch.load("clip_classifier_checkpoint.pth", map_location=device)
clip_model.load_state_dict(checkpoint["clip_model_state_dict"])
classifier.load_state_dict(checkpoint["classifier_state_dict"])

processor = CLIPProcessor.from_pretrained(model_name)

# -------------------------------------------------
# Updated HTML with image preview BEFORE upload
# -------------------------------------------------
HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Movie Genre Classifier</title>
    <style>
        body { font-family: Arial; max-width: 800px; margin: auto; }
        img { max-width: 300px; margin-top: 20px; border-radius: 10px; }
        .genres { margin-top: 20px; font-size: 1.2em; }
    </style>
</head>
<body>
    <h2>🎬 Movie Poster Genre Classifier</h2>

    <form method="post" enctype="multipart/form-data">
        <input type="file" name="poster" accept="image/*" onchange="previewImage(event)" required>
        <br><br>
        <img id="preview" style="display:none;">
        <br><br>
        <button type="submit">Predict</button>
    </form>

    {% if image_url %}
    <h3>Predicted Poster:</h3>
    <img src="{{ image_url }}">
    <div class="genres">
        <h3>Predicted Genres:</h3>
        <ul>
            {% for g in genres %}
            <li>{{ g }}</li>
            {% endfor %}
        </ul>
    </div>
    {% endif %}

<script>
function previewImage(event) {
    const img = document.getElementById("preview");
    img.src = URL.createObjectURL(event.target.files[0]);
    img.style.display = "block";
}
</script>

</body>
</html>
"""

# -------------------------------------------------
# Prediction function
# -------------------------------------------------
def predict_genres(image_path):
    image = Image.open(image_path).convert("RGB")
    inputs = processor(images=[image], return_tensors="pt").to(device)

    with torch.no_grad():
        features = clip_model.get_image_features(**inputs)
        logits = classifier(features)
        probs = torch.sigmoid(logits)

    threshold = 0.5
    idx = (probs > threshold).nonzero(as_tuple=True)[1].tolist()

    return [id_to_genre[i] for i in idx]

# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        file = request.files["poster"]
        path = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(path)

        genres = predict_genres(path)
        image_url = "/uploads/" + file.filename

        return render_template_string(HTML, image_url=image_url, genres=genres)

    return render_template_string(HTML, image_url=None, genres=None)

@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory("uploads", filename)

if __name__ == "__main__":
    app.run(debug=True)
