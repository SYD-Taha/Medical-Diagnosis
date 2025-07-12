from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from torchvision import models, transforms
from PIL import Image
import io

app = FastAPI(title="Medical Diagnosis API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Common transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

# Label maps
pneumonia_labels = ['Normal', 'Pneumonia']
tumor_labels = ['No Tumor', 'Tumor']

# Load models
def load_model(model_path):
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

pneumonia_model = load_model("D:\\Ai-lab\\portfolio Projects\\Medical Diagnosis\\models\\pneumonia_model.pt")
tumor_model = load_model("D:\\Ai-lab\\portfolio Projects\\Medical Diagnosis\\models\\tumor_model.pt")

# Prediction function
def predict(model, image_tensor):
    with torch.no_grad():
        output = model(image_tensor.unsqueeze(0).to(device))
        probs = torch.nn.functional.softmax(output[0], dim=0)
        pred_class = probs.argmax().item()
        confidence = probs[pred_class].item()
    return pred_class, confidence

# Image loading
def read_imagefile(file) -> torch.Tensor:
    image = Image.open(io.BytesIO(file)).convert("RGB")
    return transform(image)

@app.post("/predict/{diagnosis_type}")
async def predict_diagnosis(diagnosis_type: str, file: UploadFile = File(...)):
    if diagnosis_type not in ["pneumonia", "tumor"]:
        raise HTTPException(status_code=400, detail="Invalid diagnosis type")

    try:
        image_bytes = await file.read()
        img_tensor = read_imagefile(image_bytes)

        if diagnosis_type == "pneumonia":
            pred, conf = predict(pneumonia_model, img_tensor)
            label = pneumonia_labels[pred]
        else:
            pred, conf = predict(tumor_model, img_tensor)
            label = tumor_labels[pred]

        return {
            "prediction": label,
            "confidence": round(conf * 100, 2)
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
