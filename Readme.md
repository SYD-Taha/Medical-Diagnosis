
# 🧠 AI-Powered Medical Diagnosis System

This project is a **Deep Learning-based diagnostic system** that can detect:

- **Pneumonia** from Chest X-ray images
- **Brain Tumors** from MRI scans

It consists of:
- 🧠 Two trained **ResNet18 CNN models**
- ⚙️ A **FastAPI** backend for inference
- 🎨 A **Streamlit** UI for uploading images and receiving predictions
- 🔍 Grad-CAM explainability to visualize model focus (optional in backend)

---

## 🚀 Demo

https://github.com/SYD-Taha/Medical-Diagnosis

---

## 📁 Project Structure

```
diagnosis_system/
├── api/                  # FastAPI backend
│   └── main.py
├── frontend/             # Streamlit frontend
│   └── app.py
├── models/               # Trained PyTorch models
│   ├── pneumonia_model.pt
│   └── tumor_model.pt
├── data/                 # Dataset folders (excluded in .gitignore)
│   ├── chest_xray/
│   └── brain_tumor/
├── notebooks/            # Model training & EDA notebooks
│   └── day1_day2_combined.ipynb
├── requirements.txt
└── README.md
```

---

## 📦 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/ai-medical-diagnosis.git
cd ai-medical-diagnosis
```

### 2. Install Requirements

```bash
pip install -r requirements.txt
```

### 3. Download Datasets

- **Pneumonia (Chest X-ray)**  
  [Kaggle Link](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

- **Brain Tumor (MRI)**  
  [Kaggle Link](https://www.kaggle.com/datasets/navoneel/brain-mri-images-for-brain-tumor-detection)

Place them like this:

```
data/
├── chest_xray/
│   ├── train/NORMAL, PNEUMONIA
│   ├── test/NORMAL, PNEUMONIA
│   └── val/NORMAL, PNEUMONIA
├── brain_tumor/
│   ├── yes/
│   └── no/
```

---

## 🏋️‍♂️ Train the Models (Optional)

Run the notebook to train both models:

```bash
notebooks/day1_day2_combined.ipynb
```

Trained models are saved in `models/` as:
- `pneumonia_model.pt`
- `tumor_model.pt`

---

## 🔌 Run the Backend (FastAPI)

```bash
uvicorn api.main:app --reload
```

API is available at:  
[http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

## 🎨 Run the Frontend (Streamlit)

```bash
streamlit run frontend/app.py
```

You’ll see:
- Upload image
- Select diagnosis type (Pneumonia / Brain Tumor)
- View prediction + confidence

---

## 📈 Features

| Feature               | Included |
|------------------------|----------|
| Pneumonia Detection    | ✅        |
| Brain Tumor Detection  | ✅        |
| Transfer Learning      | ✅        |
| Streamlit UI           | ✅        |
| FastAPI Backend        | ✅        |
| Grad-CAM Visualization | ✅ (backend only) |

---

## 📸 Sample Prediction Output

```json
{
  "prediction": "Pneumonia",
  "confidence": 98.35
}
```

---

## 🛠️ Tech Stack

- PyTorch
- FastAPI
- Streamlit
- Torchvision
- PIL / OpenCV
- scikit-learn
- torchcam (Grad-CAM)

---

## 🙌 Contributors

- [Syed Taha Jameel](https://github.com/SYD-Taha)

---

## 📜 License

This project is licensed under the MIT License.
