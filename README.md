# Livestock Breed ID

Full-stack image classification app: YOLOv8 detects and crops the main animal, then a fine-tuned MobileNetV2 predicts its breed.

## Run

Start the API:

```powershell
cd backend
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
uvicorn app.main:app --reload --port 8000
```

Start the web app:

```powershell
cd frontend
npm install
npm run dev
```

Before starting the API, save your trained TensorFlow/Keras model at `backend/models/breed_classifier.keras` and its class names as a JSON string array at `backend/models/breed_labels.json`.

`POST /api/v1/predict` accepts up to 10 images through the multipart `files` field. Open `/docs` for interactive API docs.

## Training data

```text
data/
  train/Gir/
  train/Holstein Friesian/
  validation/Gir/
  validation/Holstein Friesian/
```

Run `python training/train_classifier.py --data-dir data --output backend/models/breed_classifier.keras` after installing the backend requirements.
